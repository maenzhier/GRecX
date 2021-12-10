#include <stdio.h>
#include <math.h>



double dcg_score(int* rels, int num){
	double score = 0.0;
	for(int i=0; i<num; i++){
		score += (pow(2.0, rels[i]) - 1.0) / log(i + 2);
	}
	// printf("s: %f\n", score);
	return score;
}

double ndcg_score(int* rels_true, int* rels_pred, int num_rels){
	return dcg_score(rels_pred, num_rels) / dcg_score(rels_true, num_rels);
}



int main(){


	int rels[] = {0, 1, 0, 0, 1, 1};
	printf("%f\n", dcg_score(rels, 6));


	int rels_true[] = {1, 1, 1, 0, 0, 0};
	int rels_pred[] = {0, 1, 0, 0, 1, 1};
	printf("%f\n", ndcg_score(rels_true, rels_pred, 6));

    return 0;
}

