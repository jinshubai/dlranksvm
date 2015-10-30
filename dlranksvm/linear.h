#ifndef _LIBLINEAR_H
#define _LIBLINEAR_H

#include <mpi.h>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

struct id_and_value
{
	int id;
	double value;
};

struct feature_node
{
	int index;
	double value;
};

struct problem
{
	int l, n;
	int *query;
	double *y;
	struct feature_node **x;
};

enum {SELECTION_TREE}; /* solver_type */

struct parameter
{
	int solver_type;

	/* these are for training only */
	double eps;	        /* stopping criteria */
	double C;
};

struct model
{
	struct parameter param;
	int nr_class;		/* number of classes */
	int nr_feature;
	double *w;
};

struct model* train(const struct problem *prob, const struct parameter *param);

double predict_values(const struct model *model_, const struct feature_node *x, double* dec_values);
double predict(const struct model *model_, const struct feature_node *x);

int save_model(const char *model_file_name, const struct model *model_);
struct model *load_model(const char *model_file_name);

int get_nr_feature(const struct model *model_);
int get_nr_class(const struct model *model_);
void free_model_content(struct model *model_ptr);
void free_and_destroy_model(struct model **model_ptr_ptr);

const char *check_parameter(const struct problem *prob, const struct parameter *param);
void set_print_string_function(void (*print_func) (const char*));
void eval_list(double *label, double *target, int *query, int l, double *result_ret);
#ifdef FIGURE56
void evaluate_test(double* w);
extern struct feature_node *x_spacetest;
extern struct problem probtest;
#endif

#ifdef __cplusplus
}
#endif

int mpi_get_rank();

int mpi_get_size();

template<typename T>
void mpi_allreduce(T *buf, const int count, MPI_Datatype type, MPI_Op op)
{
  	std::vector<T> buf_reduced(count);
	MPI_Allreduce(buf, buf_reduced.data(), count, type, op, MPI_COMM_WORLD);
	for(int i=0;i<count;i++)
		buf[i] = buf_reduced[i];
}

void mpi_exit(const int status);
#endif /* _LIBLINEAR_H */

