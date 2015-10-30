#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "linear.h"

int print_null(const char *s,...) {return 0;}

static int (*info)(const char *fmt,...) = &printf;

struct feature_node *x;
int max_nr_attr = 64;

struct model* model_;

void exit_input_error(int line_num)
{
	fprintf(stderr,"[rank %d] Wrong input format at line %d\n", mpi_get_rank(), line_num);
	mpi_exit(1);
}

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
	int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

void do_predict(FILE *input, FILE *output)
{
	int total=0;
	int n;
	int nr_feature=get_nr_feature(model_);
	double *dvec_t;
	double *ivec_t;
	int *query;
	n=nr_feature;

	max_line_len = 1024;
	line = (char *)malloc(max_line_len*sizeof(char));
	while(readline(input) != NULL)
		total++;
	rewind(input);
	dvec_t = new double[total];
	ivec_t = new double[total];
	query = new int[total];
	total = 0;
	while(readline(input) != NULL)
	{
		int i = 0;
		double target_label, predict_label;
		char *idx, *val, *label, *endptr;
		int inst_max_index = 0; // strtol gives 0 if wrong format

		query[total] = 0;
		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(total+1);

		target_label = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(total+1);
		ivec_t[total] = target_label;

		while(1)
		{
			if(i>=max_nr_attr-2)	// need one more for index = -1
			{
				max_nr_attr *= 2;
				x = (struct feature_node *) realloc(x,max_nr_attr*sizeof(struct feature_node));
			}

			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			if (strcmp(idx,"qid") == 0)
			{
				errno = 0;
				query[total] = (int) strtol(val,&endptr,10);
				if(endptr == val || errno != 0 || *endptr != '\0')
					exit_input_error(i+1);
				continue;
			}
			errno = 0;
			x[i].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x[i].index <= inst_max_index)
				exit_input_error(total+1);
			else
				inst_max_index = x[i].index;

			errno = 0;
			x[i].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(total+1);

			// feature indices larger than those in training are not used
			if(x[i].index <= nr_feature)
				++i;
		}
		x[i].index = -1;

		predict_label = predict(model_,x);
		fprintf(output,"%.32f\n",predict_label);
		dvec_t[total++] = predict_label;
	}
	
	/************************/
	int g_total = total;
	int size = mpi_get_size();
	int *send_counts;
	send_counts = new int[size];
	MPI_Allgather((void*)&total, 1, MPI_INT, (void*)send_counts, 1, MPI_INT, MPI_COMM_WORLD);

	int *recv_counts;
	recv_counts = new int[size];
	int *recv_displs;
	recv_displs = new int[size];

	for(int j=0;j<size;j++)
	{
		recv_displs[j] = 0;
		for(int k=0;k<j;k++)
		{
			recv_displs[j] += send_counts[k];
		}
		recv_counts[j] = send_counts[j]; 
	}

	mpi_allreduce(&g_total, 1, MPI_INT, MPI_SUM);
	double *g_ivec_t;
	double *g_dvec_t;
	int *g_query;
	g_ivec_t = new double[g_total];
	g_dvec_t = new double[g_total];
	g_query = new int[g_total];

	MPI_Allgatherv((void*)ivec_t, total, MPI_DOUBLE, (void*)g_ivec_t, recv_counts, recv_displs, MPI_DOUBLE, MPI_COMM_WORLD);
	MPI_Allgatherv((void*)dvec_t, total, MPI_DOUBLE, (void*)g_dvec_t, recv_counts, recv_displs, MPI_DOUBLE, MPI_COMM_WORLD);
	MPI_Allgatherv((void*)query, total, MPI_INT, (void*)g_query, recv_counts, recv_displs, MPI_INT, MPI_COMM_WORLD);
	/*******************/

	double result[3];
	eval_list(g_ivec_t,g_dvec_t,g_query,g_total,result);

	if(mpi_get_rank()==0)
	{
		info("Pairwise Accuracy = %g%%\n",result[0]*100);
		info("MeanNDCG (LETOR) = %g\n",result[1]);
		info("NDCG (YAHOO) = %g\n",result[2]);
	}

	delete ivec_t;
	delete dvec_t;
	delete query;
	delete g_ivec_t;
	delete g_dvec_t;
	delete g_query;
	delete recv_counts;
	delete recv_displs;
	delete send_counts;
}

void exit_with_help()
{ 
	if(mpi_get_rank() != 0)
		mpi_exit(1);
	printf(
	"Usage: predict [options] test_file model_file output_file\n"
	"options:\n"
	"-q : quiet mode (no outputs)\n"
	);
	mpi_exit(1);
}

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
	
	FILE *input, *output;
	int i;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		++i;
		switch(argv[i-1][1])
		{
			case 'q':
				info = &print_null;
				i--;
				break;
			default:
				fprintf(stderr,"unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
				break;
		}
	}
	if(i>=argc)
		exit_with_help();

	input = fopen(argv[i],"r");
	if(input == NULL)
	{
		fprintf(stderr,"[rank %d] can't open input file %s\n", mpi_get_rank(), argv[i]);
		//Writes the C string pointed by format to the stream
		mpi_exit(1);
	}

	output = fopen(argv[i+2],"w");
	if(output == NULL)
	{
		fprintf(stderr,"[rank %d] can't open output file %s\n", mpi_get_rank(), argv[i+2]);
		mpi_exit(1);
	}

	if((model_=load_model(argv[i+1]))==0)
	{
		fprintf(stderr,"[rank %d] can't open model file %s\n", mpi_get_rank(), argv[i+1]);
		mpi_exit(1);
	}

	x = (struct feature_node *) malloc(max_nr_attr*sizeof(struct feature_node));
	do_predict(input, output);
	free_and_destroy_model(&model_);
	free(line);
	free(x);
	fclose(input);
	fclose(output);

	MPI_Finalize();
	return 0;
}

