//#############################################################################
//#
//# NOTE: Code is heavily unoptimized with regard to memory usage,
//#       and it also contains redundant operations accumulated
//#       due to testing and experimenting with various stuff.
//#
//# Author: Nikos Komodakis
//#
//#############################################################################
#include <stdio.h>

#ifndef __CV_CLUSTERING_H__
#define __CV_CLUSTERING_H__

#include <stdlib.h>
#include <assert.h>
#include <list>

#define INF 10000000000
#define MAX(a,b) ((a)>=(b) ? (a):(b))
#define ABS(a) ((a)>=0 ? (a):-(a))

#ifdef _DO_DEBUG_
#define ASSERT(x) assert(x)
#else
#define ASSERT(x) 
#endif

//#############################################################################
//#
//# Classes & types
//#
//#############################################################################

//=============================================================================
// @author Nikos Komodakis
//=============================================================================
//
class CV_Clustering 
{

	public: // member functions

		typedef float Real;

		struct Info
		{
			int id;
			Real avg;
			Real resources;
			Real demand;

			Real *hh, *ub;
		};

		/*
		 *
		 * "dist" contains distances only between different p,q
		 *
		 */
		CV_Clustering( int num_pairs, int *p, int *q, Real *dist, int num_points, Real *penalty )
		{
			_num_pairs = num_pairs;
			_p = p;
			_q = q;
			_dist = dist;
			_num_points = num_points;
			_penalty = penalty;

			//
			//
			_fixed_height = new Real[_num_points]; 
			for( int i = 0; i < _num_points; i++ )
				_fixed_height[i] = INF;

			//
			//
			_h = new Real[_num_pairs];
			_hh = new Real[_num_points];
			_num_nbrs_for_p = new int[_num_points]; // These are actually the numbers of DIFFERENT nbrs for p
			_num_nbrs_for_q = new int[_num_points]; // These are actually the numbers of DIFFERENT nbrs for q
			for( int i = 0; i < _num_points; i++ )
			{
				_hh[i] = _penalty[i]; // I assume that dual vars are initially zero 
				_num_nbrs_for_p[i] = 0;
				_num_nbrs_for_q[i] = 0;
			}
			for( int i = 0; i < _num_pairs; i++ )
			{
				_h[i] = _dist[i]; // I assume that dual vars are initially zero 
				_num_nbrs_for_p[_p[i]]++;
				_num_nbrs_for_q[_q[i]]++;
			}

			//
			//
			_max_num_nbrs_for_p = -1;
			_max_num_nbrs_for_q = -1;

			_min_h = new Real[_num_points];
			_min_q = new int [_num_points];

			_nbrs_for_p = new int *[_num_points];
			_nbrs_for_q = new int *[_num_points];

			for( int i = 0; i < _num_points; i++ )
			{
				if ( _num_nbrs_for_p[i] > _max_num_nbrs_for_p )
					_max_num_nbrs_for_p = _num_nbrs_for_p[i];
				if ( _num_nbrs_for_q[i] > _max_num_nbrs_for_q )
					_max_num_nbrs_for_q = _num_nbrs_for_q[i];

				_nbrs_for_p[i] = new int[_num_nbrs_for_p[i]];
				_nbrs_for_q[i] = new int[_num_nbrs_for_q[i]];
				_num_nbrs_for_p[i] = 0;
				_num_nbrs_for_q[i] = 0;

				_min_h[i] = INF;
				_min_q[i] = -1;
			}
			assert( _max_num_nbrs_for_p >= 0 );
			assert( _max_num_nbrs_for_q >= 0 );

			for( int i = 0; i < _num_pairs; i++ )
			{
				int p = _p[i]; 
				int q = _q[i];
				_nbrs_for_p[p][_num_nbrs_for_p[p]++] = i; 
				_nbrs_for_q[q][_num_nbrs_for_q[q]++] = i; 

				if ( _h[i] < _min_h[p] ) 
				{
					_min_h[p] = _h[i];
					_min_q[p] = q;
				}
			}

			_cur_dual = 0;
			_pmin = new Real[_num_points]; // partial minimum
			for( int p = 0; p < _num_points; p++ )
			{
				ASSERT( _min_q[p] >= 0 );
				_pmin[p] = _min_h[p];
				if ( _hh[p] < _min_h[p] )
				{
					_min_h[p] = _hh[p];
					_min_q[p] = p;
				}

				_cur_dual += _min_h[p];
			}

			//
			//
			_exemplars = new int[_num_points];
			_min_d = new Real[_num_points];

			_avg = new Real[_num_points]; 
			_avg2 = new Real[_num_points]; 
			_info = new Info[2]; 
			_num_min = new int[_num_points]; 
			_ppmin = new Real[_num_points]; 

			_best_exemplars = _exemplars;

			_assigned = new int[_num_points];

			_flabel = new int[_num_points];
			_resou = new Real[_num_points]; 
			_resou2 = new Real[_num_points]; 

			_special = new int[_num_points]; 
		}

		virtual ~CV_Clustering( void )
		{
			delete [] _h;
			delete [] _hh;

			delete [] _num_nbrs_for_p;
			delete [] _num_nbrs_for_q;

			for( int i = 0; i < _num_points; i++ )
			{
				delete [] _nbrs_for_p[i];
				delete [] _nbrs_for_q[i];
			}
			delete [] _nbrs_for_p;
			delete [] _nbrs_for_q;

			delete [] _min_h;
			delete [] _min_q;

			delete [] _exemplars;
			delete [] _min_d;

			delete [] _avg;
			delete [] _avg2;
			delete [] _info;
			delete [] _pmin;
			delete [] _num_min;
			delete [] _ppmin;

			delete [] _assigned;

			delete [] _flabel;
			delete [] _resou;
			delete [] _resou2;

			delete [] _special;
			delete [] _fixed_height;
		}

		void run( int max_iters )
		{
			_cur_primal = -1;

			_num_exemplars = 0;
			for( int q = 0; q < _num_points; q++ )
			{
				_assigned[q] = 0;
				_flabel[q] = -1;
			}

			_best_primal = INF;
			_best_num_exemplars = -1;

			_dual_function.clear();
			_primal_function.clear();

			int wait = 0;

			while (1)
			{
				for( int it = 0; it < max_iters; it++ )
				{
					Real tmp_dual = _cur_dual;

					int num_zero_ah = 0;
	
					if ( !wait )
						compute_demands();

					for( int q = 0; q < _num_points; q++ )
					{
						if ( _assigned[q] )
							continue;

						_resou[q] = _resou2[q] - _avg2[q];
						
						if ( _num_min[q] == 0 )
						{
							_resou[q] = -1;
							_avg[q] = -1;
						}
						else
						{
							_resou[q] /= _num_min[q];
							_avg[q] = -_resou[q];
						}

						if ( _avg[q] == 0 )
							num_zero_ah++;
					}

					Real maxval = -INF;
					Real bestq = -1;
					for( int q = 0; q < _num_points; q++ )
					{
						if ( !_assigned[q] )
						{
							Real d = _resou[q];
							if ( d > maxval )
							{
								maxval = d;
								bestq = q;
							}
						}
					}
					_info[0].resources = maxval;
					_info[0].demand = 0;
					_info[0].id = bestq;

					if ( maxval < 0 )
					{
						wait = 0;
						for( int i = 0; i < _num_pairs; i++ )
						{
							int  p = _p[i];
							int  q = _q[i];

							if ( _assigned[p] || _assigned[q] )
								continue;
							assert( _avg[q] >= 0 );

							if ( _min_h[p] >= _dist[i] )
							{
								if ( _min_h[p] < _fixed_height[p] )
								{
									if ( _h[i] == _min_h[p] )
										_h[i] = _ppmin[p] + _avg[q];
									else
										_h[i] = _min_h[p] + _avg[q];
								}
								else if ( _avg[q] >= 0 )  
								{
									_h[i] = _min_h[p];
									assert( _min_h[p] == _fixed_height[p] );
								}
							}
							else if ( _avg[q] >= 0 )
								_h[i] = _dist[i];  

							assert( _h[i] >= _dist[i] );
						}
						for( int q = 0; q < _num_points; q++ )
						{
							if ( _assigned[q] )
								continue;
							assert( _avg[q] >= 0 );

							if ( _min_h[q] >= _fixed_height[q] && _special[q] )
							{
								_hh[q] = _min_h[q];
								assert( _min_h[q] == _fixed_height[q] );
								continue;
							}

							if ( _avg[q] >= 0 )
								if ( _hh[q] == _min_h[q] )
									_hh[q] = _ppmin[q] + _avg[q];
								else
									_hh[q] = _min_h[q] + _avg[q];
						}
					}
					else
					{
						wait = 1;
						break;
					}

					//
					//
					for( int q = 0; q < _num_points; q++ )
					{
						_min_h[q] = INF;
						_min_q[q] = -1;
					}
	
					for( int i = 0; i < _num_pairs; i++ )
					{
						int  p = _p[i];
						int  q = _q[i];
	
						if ( _h[i] < _min_h[p] )
						{
							_min_h[p] = _h[i];
							_min_q[p] = q;
						}
						else if ( _h[i] == _min_h[p] && _assigned[q] )
						{
							_min_h[p] = _h[i];
							_min_q[p] = q;
						}
					}
					_cur_dual = 0;
					for( int q = 0; q < _num_points; q++ )
					{
						ASSERT( _min_q[q] >= 0 );
						_pmin[q] = _min_h[q];
						if ( _hh[q] < _min_h[q] )
						{
							_min_h[q] = _hh[q];
							_min_q[q] = q;
						}
						else if ( _hh[q] == _min_h[q] && _assigned[q] )
						{
							_min_h[q] = _hh[q];
							_min_q[q] = q;
						}
						_cur_dual += _min_h[q];
					}

					_primal_function.push_back( _cur_primal );
					_dual_function.push_back( _cur_dual );

					_best_primal = _cur_primal;
					_best_num_exemplars = _num_exemplars;
	
					if ( ABS(_cur_dual-tmp_dual) == 0 )
						break;
				}

				if ( !add_exemplar() )
					break;
				compute_primal();
			}

			compute_primal2();
			printf( "Converged (%d clusters)\n", _num_exemplars );
		}

		int add_exemplar( void )
		{
			int new_min = 0;
			int next = 0;

			Real val = _info[next].resources - _info[next].demand;
			if ( val < 0 )
				return 0;

			int next_exemplar = _info[next].id;

			int p = next_exemplar;
			assert( !_assigned[p] );

			_assigned[p] = 1;
			printf( "Adding datapoint %d as cluster center\n", p );
			_exemplars[_num_exemplars++] = p;
			_flabel[p] = _num_exemplars-1;

			p = next_exemplar;
			int *nbrs_for_p = _nbrs_for_p[p];
			for( int k = 0; k < _num_nbrs_for_p[p]; k++ )
			{
				int i = nbrs_for_p[k];
				int qq = _q[i];

				if ( _assigned[qq] ) continue;

				Real prev_avg = _h[i] - MAX( _min_h[p], _dist[i] );
				Real prev_res = ( _h[i] == _min_h[p] && _min_h[p] < _fixed_height[p] ?  _ppmin[p] - _min_h[p] : 0 );
				int  prev_num_min = ( _min_h[p] >= _dist[i] && _min_h[p] < _fixed_height[p] ); 

				_avg2[qq]   += - prev_avg;
				_resou2[qq] += - prev_res;
				_num_min[qq] += - prev_num_min;
			}

			int mask;

			int q = next_exemplar;
			int *nbrs_for_q = _nbrs_for_q[q];
			for( int k = 0; k < _num_nbrs_for_q[q]; k++ )
			{
				int i = nbrs_for_q[k];
				int p = _p[i];

				if ( !_assigned[p] && _h[i] == _min_h[p] && _h[i] > _dist[i] )
					_flabel[p] = _num_exemplars-1;
				Real delta = _h[i] - _dist[i];
				if ( delta < 0 )
					delta = 0;

				if ( _h[i]-delta < _ppmin[p] )
				{
					Real new_min_h, new_ppmin;
					if ( _h[i]-delta < _min_h[p] )
					{
						new_min_h = _h[i]-delta;
						if ( _min_q[p] != q )
							new_ppmin = _min_h[p];
						else
							new_ppmin = _ppmin[p];
					}
					else 
					{
						new_min_h = _min_h[p];
						if ( _min_q[p] != q )
							new_ppmin = _h[i]-delta;
						else
							new_ppmin = _ppmin[p];
					}

					Real new_fixed_height = ( _h[i] - delta < _fixed_height[p] ? _h[i] - delta : _fixed_height[p] );

					if ( !_assigned[p] )
					{
						int *nbrs_for_p = _nbrs_for_p[p];
						for( int kk = 0; kk < _num_nbrs_for_p[p]; kk++ )
						{
							int i = nbrs_for_p[kk];
							int qq = _q[i];
	
							if ( qq == q ) mask = 0;
							else if ( _assigned[qq] ) continue;
							else mask = 1;
	
							Real prev_avg = _h[i] - MAX( _min_h[p], _dist[i] );
							Real new_avg  = _h[i] - MAX( new_min_h, _dist[i] ) * mask;
	
							Real prev_res = ( _h[i] == _min_h[p] && _min_h[p] < _fixed_height[p] ?  _ppmin[p] - _min_h[p] : 0 );
							Real new_res  = ( _h[i] == new_min_h && new_min_h < new_fixed_height ?  new_ppmin - new_min_h : 0 ) * mask;

							int  prev_num_min = ( _min_h[p] >= _dist[i] && _min_h[p] < _fixed_height[p] ); 
							int  new_num_min  = ( new_min_h >= _dist[i] && new_min_h < new_fixed_height ) * mask; 

							_avg2[qq]   += new_avg - prev_avg;
							_resou2[qq] += new_res - prev_res;
							_num_min[qq] += new_num_min - prev_num_min;
						}
						Real prev_avg = _hh[p] - _min_h[p];
						Real new_avg  = _hh[p] - new_min_h;
						Real prev_res = ( _hh[p] == _min_h[p] && _min_h[p] < _fixed_height[p] ?  _ppmin[p] - _min_h[p] : 0 );
						Real new_res  = ( _hh[p] == new_min_h && new_min_h < new_fixed_height ?  new_ppmin - new_min_h : 0 );
						_avg2[p]   += new_avg - prev_avg;
						_resou2[p] += new_res - prev_res;
	
						int  prev_num_min = ( _min_h[p] < _fixed_height[p] || _special[p] == 0 ); 
					}

					_min_h[p] = new_min_h;
					_ppmin[p] = new_ppmin;
				}

				_h[i] -= delta;
				if ( _h[i] <= _min_h[p] )
					_min_q[p] = q;
				if ( _h[i] < _fixed_height[p] )
					_fixed_height[p] = _h[i];
				if ( _h[i] <= _pmin[p] )
					_pmin[p] = _h[i];
				_hh[q] += delta;
			}

			p = next_exemplar;
			nbrs_for_p = _nbrs_for_p[p];
			for( int k = 0; k < _num_nbrs_for_p[p]; k++ )
			{
				int i = nbrs_for_p[k];
				int q = _q[i];

				Real delta = _h[i] - _dist[i];
				if ( delta < 0 )
					delta = 0;

				_h[i] -= delta;

				if ( _hh[q] <= _ppmin[q] )
				{
					int pp = q;

					Real new_hh = _hh[pp]+delta;
					Real new_min_h, new_ppmin, new_pmin;
					int new_min_q;
					update_min2( pp, new_hh, &new_min_h, &new_ppmin, &new_pmin, &new_min_q );

					Real new_fixed_height = _fixed_height[pp];

					if ( !_assigned[pp] )
					{
						int *nbrs_for_p = _nbrs_for_p[pp];
						for( int kk = 0; kk < _num_nbrs_for_p[pp]; kk++ )
						{
							int i = nbrs_for_p[kk];
							int qq = _q[i];
	
							if ( qq == p ) mask = 0;
							else if ( _assigned[qq] ) continue;
							else mask = 1;
	
							Real prev_avg = _h[i] - MAX( _min_h[pp], _dist[i] );
							Real new_avg  = _h[i] - MAX( new_min_h, _dist[i] ) * mask;
	
							Real prev_res = ( _h[i] == _min_h[pp] && _min_h[pp] < _fixed_height[pp] ?  _ppmin[pp] - _min_h[pp] : 0 );
							Real new_res  = ( _h[i] == new_min_h && new_min_h < new_fixed_height ?  new_ppmin - new_min_h : 0 ) * mask;
	
							int  prev_num_min = ( _min_h[pp] >= _dist[i] && _min_h[pp] < _fixed_height[pp] ); 
							int  new_num_min = ( new_min_h >= _dist[i] && new_min_h < new_fixed_height ) * mask; 
	
							_avg2[qq]   += new_avg - prev_avg;
							_resou2[qq] += new_res - prev_res;
							_num_min[pp] += new_num_min - prev_num_min;
						}
						Real prev_avg = _hh[pp] - _min_h[pp];
						Real new_avg  = new_hh - new_min_h;
						Real prev_res = ( _hh[pp] == _min_h[pp] && _min_h[pp] < _fixed_height[pp] ?  _ppmin[pp] - _min_h[pp] : 0 );
						Real new_res  = ( new_hh == new_min_h && new_min_h < new_fixed_height ?  new_ppmin - new_min_h : 0 );
						_avg2[pp]   += new_avg - prev_avg;
						_resou2[pp] += new_res - prev_res;
	
						int  prev_num_min = ( _min_h[pp] < _fixed_height[pp] || _special[pp] == 0 ); 
					}

					_min_h[pp] = new_min_h;
					_ppmin[pp] = new_ppmin;
					_min_q[pp] = new_min_q;
					_pmin[pp]  = new_pmin;
				}
				else
				{
					if ( !_assigned[q] ) 
						_avg2[q] += delta;
				}

				_hh[q] += delta;

			}

			return 1;
		}

		void update_min2( int p, Real new_hh, Real *new_min_h, Real *new_ppmin, Real *new_pmin, int *new_min_q )
		{
			int *nbrs_for_p = _nbrs_for_p[p];
			Real min_val = INF;
			int min_q = -1;
			for( int k = 0; k < _num_nbrs_for_p[p]; k++ )
			{
				int i = nbrs_for_p[k];
				if ( _h[i] < min_val )
				{
					min_val = _h[i];
					min_q = _q[i];
				}
			}

			*new_pmin = min_val; 

			assert( min_q >= 0 );
			if ( new_hh < min_val )
			{
				min_val = new_hh;
				min_q = p;
			}

			*new_min_h = min_val;
			*new_min_q = min_q;

			Real ppmin = INF;
			for( int k = 0; k < _num_nbrs_for_p[p]; k++ )
			{
				int i = nbrs_for_p[k];
				int q = _q[i];
				if ( q != min_q && _h[i] < ppmin )
					ppmin = _h[i];
			}
			if ( p != min_q && new_hh < ppmin )
				ppmin = new_hh;
			*new_ppmin = ppmin;
		}

		void update_min( int p )
		{
			int *nbrs_for_p = _nbrs_for_p[p];
			Real min_val = INF;
			int min_q = -1;
			for( int k = 0; k < _num_nbrs_for_p[p]; k++ )
			{
				int i = nbrs_for_p[k];
				if ( _h[i] < min_val )
				{
					min_val = _h[i];
					min_q = _q[i];
				}
			}

			_pmin[p] = min_val; 

			assert( min_q >= 0 );
			if ( _hh[p] < min_val )
			{
				min_val = _hh[p];
				min_q = p;
			}

			_cur_dual += min_val - _min_h[p];
			_min_h[p] = min_val;
			_min_q[p] = min_q;

			Real ppmin = INF;
			for( int k = 0; k < _num_nbrs_for_p[p]; k++ )
			{
				int i = nbrs_for_p[k];
				int q = _q[i];
				if ( q != _min_q[p] && _h[i] < ppmin )
					ppmin = _h[i];
			}
			if ( p != _min_q[p] && _hh[p] < ppmin )
				ppmin = _hh[p];
			_ppmin[p] = ppmin;
		}

		void compute_primal( void )
		{
			for( int p = 0; p < _num_points; p++ )
			{
				_min_d[p] = INF;
				//_label[p] = -1;
			}

			for( int e = 0; e < _num_exemplars; e++ )
			{
				int q = _exemplars[e];
				int *nbrs_for_q = _nbrs_for_q[q];
				for( int k = 0; k < _num_nbrs_for_q[q]; k++ )
				{
					int i = nbrs_for_q[k];
					int p = _p[i];
					if ( _dist[i] < _min_d[p] )
					{
						_min_d[p] = _dist[i];
						//_label[p] = q;
					}
				}
			}

			for( int e = 0; e < _num_exemplars; e++ )
			{
				int q = _exemplars[e];
				_min_d[q] = _penalty[q];  
			}

			int inf_count = 0;
			_cur_primal = 0;
			for( int p = 0; p < _num_points; p++ )
			{
				if ( _min_d[p] == INF )
					inf_count++;
				else
					_cur_primal += _min_d[p];
			}
		}

		void compute_primal2( void )
		{
			for( int p = 0; p < _num_points; p++ )
			{
				_min_d[p] = INF;
				//_label[p] = -1;
			}

			for( int e = 0; e < _num_exemplars; e++ )
			{
				int q = _exemplars[e];
				int *nbrs_for_q = _nbrs_for_q[q];
				for( int k = 0; k < _num_nbrs_for_q[q]; k++ )
				{
					int i = nbrs_for_q[k];
					int p = _p[i];
					if ( _dist[i] < _min_d[p] )
					{
						_min_d[p] = _dist[i];
						//_label[p] = q;
					}
				}
			}

			for( int e = 0; e < _num_exemplars; e++ )
			{
				int q = _exemplars[e];
				_min_d[q] = _penalty[q]; 
			}

			int inf_count = 0;
			_cur_primal = 0;
			for( int p = 0; p < _num_points; p++ )
			{
				_cur_primal += _min_d[p];
				if ( _min_d[p] == INF )
				{
					inf_count++;
					add_exemplar2( _min_q[p] );

					int q = _min_q[p];
					int *nbrs_for_q = _nbrs_for_q[q];
					for( int k = 0; k < _num_nbrs_for_q[q]; k++ )
					{
						int i = nbrs_for_q[k];
						int pp = _p[i];
						if ( _dist[i] < _min_d[pp] && !_assigned[pp] )
						{
							if ( pp <= p )
								_cur_primal += _dist[i] - _min_d[pp];
							_min_d[pp] = _dist[i];
							//_label[pp] = q;
						}
					}

					if ( q <= p )
						_cur_primal += _penalty[q] - _min_d[q];
					_min_d[q] = _penalty[q];
				}
			}
			_primal_function.push_back( _cur_primal );
			_dual_function.push_back( _cur_dual );
		}

		void add_exemplar2( int qq )
		{
			int p = qq; 
			assert( !_assigned[p] );

			_assigned[p] = 1;
			printf( "Adding datapoint %d as cluster center\n", p );
			_exemplars[_num_exemplars++] = p;
			_best_num_exemplars++;

			int q = qq;
			int *nbrs_for_q = _nbrs_for_q[q];
			for( int k = 0; k < _num_nbrs_for_q[q]; k++ )
			{
				int i = nbrs_for_q[k];
				int p = _p[i];

				Real delta = _h[i] - _dist[i];
				assert( delta >= 0 );

				_h[i] -= delta;
				if ( _h[i] <= _min_h[p] )
				{
					_min_h[p] = _h[i];
					_min_q[p] = q;
				}
				_hh[q] += delta;
			}
			if ( _min_q[q] == q )
				update_min( q );

			p = qq;
			int *nbrs_for_p = _nbrs_for_p[p];
			for( int k = 0; k < _num_nbrs_for_p[p]; k++ )
			{
				int i = nbrs_for_p[k];
				int q = _q[i];

				Real delta = _h[i] - _dist[i];
				assert( delta >= 0 );

				_h[i] -= delta;
				_hh[q] += delta;

				if ( _min_q[q] == q )
					update_min( q );
			}
		}

		void compute_labeling( void )
		{
			for( int p = 0; p < _num_points; p++ )
			{
				_min_d[p] = INF;
				_flabel[p] = -1;
			}

			for( int e = 0; e < _best_num_exemplars; e++ )
			{
				int q = _best_exemplars[e];
				int *nbrs_for_q = _nbrs_for_q[q];
				for( int k = 0; k < _num_nbrs_for_q[q]; k++ )
				{
					int i = nbrs_for_q[k];
					int p = _p[i];
					if ( _dist[i] < _min_d[p] )
					{
						_min_d[p] = _dist[i];
						_flabel[p] = e;
					}
				}
			}

			for( int e = 0; e < _best_num_exemplars; e++ )
			{
				int q = _best_exemplars[e];
				_min_d[q] = _penalty[q];  
				_flabel[q] = e;
			}
		}

		void compute_demands( void )
		{
			for( int q = 0; q < _num_points; q++ )
			{
				_avg2[q] = 0;
				_num_min[q] = 0;
				_resou2[q] = 0;
				_ppmin[q] = INF;
				_special[q] = 0;
			}

			for( int i = 0; i < _num_pairs; i++ )
			{
				int  p = _p[i];
				int  q = _q[i];

				if ( _min_q[p] != q && _h[i] < _ppmin[p] )
					_ppmin[p] = _h[i];
			}
			for( int q = 0; q < _num_points; q++ )
			{
				if ( _min_q[q] != q && _hh[q] < _ppmin[q] )
					_ppmin[q] = _hh[q];
			}

			for( int i = 0; i < _num_pairs; i++ )
			{
				int  p = _p[i];
				int  q = _q[i];

				if ( _assigned[p] || _assigned[q] )
					continue;

				Real delta = _h[i] - MAX( _min_h[p], _dist[i] );
				_avg2[q] += delta;
				if ( _min_h[p] >= _dist[i] && _min_h[p] < _fixed_height[p] )
					_num_min[q]++;
				ASSERT( delta >= 0 );

				if ( _h[i] == _min_h[p] && _min_h[p] < _fixed_height[p] ) 
				{
					Real delta = _ppmin[p] - _h[i];
					_resou2[q] += delta;
					assert( delta >= 0 );
				}
			}
			for( int q = 0; q < _num_points; q++ )
			{
				if ( _assigned[q] )
					continue;

				Real delta = _hh[q] - _min_h[q];
				_avg2[q] += delta;
				_num_min[q]++;
				ASSERT( delta >= 0 );

				if ( _hh[q] == _min_h[q] && _min_h[q] < _fixed_height[q] ) 
				{
					Real delta = _pmin[q] - _hh[q];
					_resou2[q] += delta;
					assert( delta >= 0 );
				}
			}
			for( int q = 0; q < _num_points; q++ )
				if ( _num_min[q] <= 0 && !_assigned[q] )
					assert(0);
		}

		int _num_points;
		int _num_pairs;
		int *_p;
		int *_q;
		Real *_dist;
		Real *_penalty;
		Real *_h;
		Real *_hh;

		int   _max_num_nbrs_for_p;
		int   _max_num_nbrs_for_q;
		int  *_num_nbrs_for_p;
		int  *_num_nbrs_for_q;
		int **_nbrs_for_p;
		int **_nbrs_for_q;

		Real *_min_h;
		int  *_min_q;

		int   _num_exemplars;
		int  *_exemplars;
		Real *_min_d;

		std::list<Real> _dual_function;
		Real _cur_dual;

		std::list<Real> _primal_function;
		Real _cur_primal;

		Real *_avg;
		Real *_avg2;
		Info *_info;
		Real *_pmin;
		int  *_num_min;
		int _K;
		Real _hK;
		int _it;
		int _found;
		Real *_ppmin;

		Real _best_primal;
		int _best_num_exemplars;
		int _total_it;
		int *_best_exemplars;

		int *_assigned;
		int *_flabel;
		int  _time;

		Real *_resou;
		Real *_resou2;

		int *_special;
		Real *_fixed_height;
};

#endif /* __CV_CLUSTERING_H__ */

//#############################################################################
//#
//# EOF
//#
//#############################################################################

