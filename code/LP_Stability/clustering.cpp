#include <stdio.h>
#include "clustering.h"

typedef CV_Clustering::Real Real;

int main(int argc, char **argv)
{
	// Read the input data
	//
	FILE *fp = fopen( argv[1], "rb" );
	if ( !fp )
	{
		printf( "Cannot open input file\n" );
		exit(1);
	}

	//
	//
	int num_pairs;
	fread( &num_pairs, sizeof(int), 1, fp );

	int *p = new int[num_pairs];
	fread( p, sizeof(int), num_pairs, fp );

	int *q = new int[num_pairs];
	fread( q, sizeof(int), num_pairs, fp );

	Real *dist = new Real[num_pairs];
	fread( dist, sizeof(Real), num_pairs, fp );

	int num_points = -1;
	for( int i = 0; i < num_pairs; i++ )
	{
		if ( p[i] > num_points )
			num_points = p[i];
		if ( q[i] > num_points )
			num_points = q[i];
	}
	num_points++;
	assert( num_points > 0 );

	Real *penalty = new Real[num_points];
	fread( penalty, sizeof(Real), num_points, fp );
	int iters;
	int num = fread( &iters, sizeof(int), 1, fp );
	assert( num == 1 );

	fclose(fp);

	// Run the algorithm
	//
	CV_Clustering clusterer( num_pairs, p, q, dist, num_points, penalty );
	clusterer.run( iters );

	// Return results
	//
	fp = fopen( argv[2], "wb" );
	if ( !fp )
	{
		printf( "Cannot create output file\n" );
		exit(1);
	}

	fwrite( &clusterer._best_num_exemplars, sizeof(int), 1, fp );
	for( int k = 0; k < clusterer._best_num_exemplars; k++ )
		fwrite( &clusterer._best_exemplars[k], sizeof(int), 1, fp );

	int tmp = clusterer._primal_function.size();
	fwrite( &tmp, sizeof(int), 1, fp );

	for( std::list<Real>::iterator it = clusterer._primal_function.begin(); it != clusterer._primal_function.end(); it++ )
	{
		Real tmp = *it;
		fwrite( &tmp, sizeof(Real), 1, fp );
	}

	for( std::list<Real>::iterator it = clusterer._dual_function.begin(); it != clusterer._dual_function.end(); it++ )
	{
		Real tmp = *it;
		fwrite( &tmp, sizeof(Real), 1, fp );
	}

	clusterer.compute_labeling();
	fwrite( clusterer._flabel, sizeof(int), clusterer._num_points, fp );

	fclose(fp);
	return 0;
}

