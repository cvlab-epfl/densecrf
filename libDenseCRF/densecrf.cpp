/*
  Copyright (C) 2016 Ecole Polytechnique Federale de Lausanne 
  Modified by Timur Bagautdinov (timur.bagautdinov@epfl.ch) 

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>

  This file incorporates work covered by the following copyright and  
  permission notice:  

  Copyright (c) 2011, Philipp Krähenbühl 
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:
  * Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
  * Neither the name of the Stanford University nor the
  names of its contributors may be used to endorse or promote products
  derived from this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY Philipp Krähenbühl ''AS IS'' AND ANY
  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL Philipp Krähenbühl BE LIABLE FOR ANY
  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <cmath>
#include <cstring>
#include <iostream>
#include <cstdlib>
#include <stdexcept>
#include <algorithm>

using std::cout;
using std::endl;
using std::string;
#include "../util/Timer.h"

#include "densecrf.h"
#include "fastmath.h"
#include "permutohedral.h"
#include "util.h"

PairwisePotential::~PairwisePotential()
{
}

SemiMetricFunction::~SemiMetricFunction()
{
}

class PottsPotential: public PairwisePotential
{
protected:
  Permutohedral lattice_;
  PottsPotential( const PottsPotential&o ){}
  int N_;
  float w_;
  float *norm_;
public:
  ~PottsPotential()
  {
    deallocate( norm_ );
  }

  PottsPotential(const float* features, int D, int N, float w, bool per_pixel_normalization=true):
    N_(N), w_(w)
  {
    lattice_.init( features, D, N );
    norm_ = allocate( N );
    for ( int i=0; i<N; i++ )
      norm_[i] = 1;
    // Compute the normalization factor
    lattice_.compute( norm_, norm_, 1 );
    if ( per_pixel_normalization ) {
      // use a per pixel normalization
      for ( int i=0; i<N; i++ )
	norm_[i] = 1.f / (norm_[i]+1e-20f);
    }
    else
    {
      float mean_norm = 0;
      for ( int i=0; i<N; i++ )
	mean_norm += norm_[i];
      mean_norm = N / mean_norm;
      // use a per pixel normalization
      for ( int i=0; i<N; i++ )
	norm_[i] = mean_norm;
    }
  }
  
  void apply(float* out_values, const float* in_values, float* tmp, int value_size) const
  {
    lattice_.compute( tmp, in_values, value_size );
    for ( int i=0,k=0; i<N_; i++ )
      for ( int j=0; j<value_size; j++, k++ )
	out_values[k] += w_*norm_[i]*tmp[k];
  }
};

class SemiMetricPotential: public PottsPotential
{
protected:
  const SemiMetricFunction * function_;
public:
  void apply(float* out_values, const float* in_values, float* tmp, int value_size) const
  {
    lattice_.compute( tmp, in_values, value_size );

    // To the metric transform
    float * tmp2 = new float[value_size];
    for ( int i=0; i<N_; i++ ) {
      float * out = out_values + i*value_size;
      float * t1  = tmp  + i*value_size;
      function_->apply( tmp2, t1, value_size );
      for ( int j=0; j<value_size; j++ )
	out[j] -= w_*norm_[i]*tmp2[j];
    }
    delete[] tmp2;
  }
  SemiMetricPotential(const float* features, int D, int N, float w,
		      const SemiMetricFunction* function, bool per_pixel_normalization=true)
    :PottsPotential( features, D, N, w, per_pixel_normalization ),function_(function)
  {}
};



/////////////////////////////
/////  Alloc / Dealloc  /////
/////////////////////////////
DenseCRF::DenseCRF(int N, int M)
  : N_(N), M_(M)
{
  unary_ = allocate( N_*M_ );
  additional_unary_ = allocate( N_*M_ );
  current_ = allocate( N_*M_ );
  next_ = allocate( N_*M_ );
  tmp_ = allocate( 2*N_*M_ );
  // Set the additional_unary_ to zero
  memset( additional_unary_, 0, sizeof(float)*N_*M_ );
}
DenseCRF::~DenseCRF()
{
  deallocate( unary_ );
  deallocate( additional_unary_ );
  deallocate( current_ );
  deallocate( next_ );
  deallocate( tmp_ );
  for( unsigned int i=0; i<pairwise_.size(); i++ )
    delete pairwise_[i];
}
DenseCRF2D::DenseCRF2D(int W, int H, int M)
  : DenseCRF(W*H,M), W_(W), H_(H)
{
}
DenseCRF2D::~DenseCRF2D()
{
}
/////////////////////////////////
/////  Pairwise Potentials  /////
/////////////////////////////////
void DenseCRF::addPairwiseEnergy (const float* features, int D, float w, const SemiMetricFunction * function)
{
  if (function)
    addPairwiseEnergy( new SemiMetricPotential( features, D, N_, w, function ) );
  else
    addPairwiseEnergy( new PottsPotential( features, D, N_, w ) );
}

void DenseCRF::addPairwiseEnergy ( PairwisePotential* potential )
{
  pairwise_.push_back( potential );
}

void DenseCRF2D::addPairwiseGaussian ( float sx, float sy, float w, const SemiMetricFunction * function )
{
  float * feature = new float [N_*2];
  for( int j=0; j<H_; j++ )
    for( int i=0; i<W_; i++ ){
      feature[(j*W_+i)*2+0] = i / sx;
      feature[(j*W_+i)*2+1] = j / sy;
    }
  addPairwiseEnergy( feature, 2, w, function );
  delete [] feature;
}

void DenseCRF2D::addPairwiseBilateral ( float sx, float sy, float sr, float sg, float sb,
					const unsigned char* im, float w, const SemiMetricFunction * function )
{
  float * feature = new float [N_*5];
  for( int j=0; j<H_; j++ )
    for( int i=0; i<W_; i++ )
    {
      feature[(j*W_+i)*5+0] = i / sx;
      feature[(j*W_+i)*5+1] = j / sy;
      feature[(j*W_+i)*5+2] = im[(i+j*W_)*3+0] / sr;
      feature[(j*W_+i)*5+3] = im[(i+j*W_)*3+1] / sg;
      feature[(j*W_+i)*5+4] = im[(i+j*W_)*3+2] / sb;
    }
  addPairwiseEnergy( feature, 5, w, function );
  delete [] feature;
}
//////////////////////////////
/////  Unary Potentials  /////
//////////////////////////////
void DenseCRF::setUnaryEnergy(const float* unary)
{
  memcpy( unary_, unary, N_*M_*sizeof(float) );
}
void DenseCRF::setUnaryEnergyForN(int n, const float* unary)
{
  memcpy( unary_+n*M_, unary, M_*sizeof(float) );
}
void DenseCRF2D::setUnaryEnergyForXY(int x, int y, const float* unary)
{
  memcpy( unary_+(x+y*W_)*M_, unary, M_*sizeof(float) );
}
///////////////////////
/////  Inference  /////
///////////////////////
void DenseCRF::inference ( int n_iterations, float* result, float relax )
{
  // Run inference
  float * prob = runInference( n_iterations, relax );
  // Copy the result over
  for( int i=0; i<N_; i++ )
    memcpy( result+i*M_, prob+i*M_, M_*sizeof(float) );
}

void DenseCRF::map ( int n_iterations, short* result, float relax )
{
  // Run inference
  float * prob = runInference( n_iterations, relax );
	
  // Find the map
  for( int i=0; i<N_; i++ ){
    const float * p = prob + i*M_;
    // Find the max and subtract it so that the exp doesn't explode
    float mx = p[0];
    int imx = 0;
    for( int j=1; j<M_; j++ )
      if( mx < p[j] ){
	mx = p[j];
	imx = j;
      }
    result[i] = imx;
  }
}

float* DenseCRF::runInference( int n_iterations, float relax )
{
  startInference();
  for( int it=0; it<n_iterations; it++ )
  {
    curr_iter_ = it;
    stepInference(relax);
  }
  return current_;
}

void DenseCRF::expAndNormalize ( float* out, const float* in, float scale, float relax )
{
  //float *V = new float[ N_+10 ];
  float *V = new float[M_];

  for( int i=0; i<N_; i++ ){
    const float * b = in + i*M_;
    // Find the max and subtract it so that the exp doesn't explode
    float mx = scale*b[0];
    for( int j=1; j<M_; j++ )
      if( mx < scale*b[j] )
	mx = scale*b[j];
    float tt = 0;
    for( int j=0; j<M_; j++ ){
      V[j] = fast_exp( scale*b[j]-mx );
      tt += V[j];
    }
    // Make it a probability
    for( int j=0; j<M_; j++ )
      V[j] /= tt;

    // smoothing in mean space
    float * a = out + i*M_;
    for( int j=0; j<M_; j++ )
      if (relax == 1)
	a[j] = V[j];
      else
	a[j] = (1-relax)*a[j] + relax*V[j];
  }
  delete[] V;
}
///////////////////
/////  Debug  /////
///////////////////

void DenseCRF::unaryEnergy(const short* ass, float* result)
{
  for( int i=0; i<N_; i++ )
    if ( 0 <= ass[i] && ass[i] < M_ )
      result[i] = unary_[ M_*i + ass[i] ];
    else
      result[i] = 0;
}

void DenseCRF::pairwiseEnergy(const short* ass, float* result, int term)
{
  float * current = allocate( N_*M_ );
  // Build the current belief [binary assignment]
  for( int i=0,k=0; i<N_; i++ )
    for( int j=0; j<M_; j++, k++ )
      current[k] = (ass[i] == j);
	
  for( int i=0; i<N_*M_; i++ )
    next_[i] = 0;
  if (term == -1)
    for( unsigned int i=0; i<pairwise_.size(); i++ )
      pairwise_[i]->apply( next_, current, tmp_, M_ );
  else
    pairwise_[ term ]->apply( next_, current, tmp_, M_ );
  for( int i=0; i<N_; i++ )
    if ( 0 <= ass[i] && ass[i] < M_ )
      result[i] =-next_[ i*M_ + ass[i] ];
    else
      result[i] = 0;
  deallocate( current );
}

void DenseCRF::startInference()
{
  // Initialize using the unary energies
  expAndNormalize( current_, unary_, -1 );
}

void DenseCRF::stepInference( float relax )
{
#ifdef SSE_DENSE_CRF
  __m128 * sse_next_ = (__m128*)next_;
  __m128 * sse_unary_ = (__m128*)unary_;
  __m128 * sse_additional_unary_ = (__m128*)additional_unary_;
#endif
  // Set the unary potential
#ifdef SSE_DENSE_CRF
  for( int i=0; i<(N_*M_-1)/4+1; i++ )
    sse_next_[i] = - sse_unary_[i] - sse_additional_unary_[i];
#else
  for( int i=0; i<N_*M_; i++ )
    next_[i] = -unary_[i] - additional_unary_[i];
#endif
	
  // Add up all pairwise potentials
  for (size_t i = 0; i < pairwise_.size(); i++)
    pairwise_[i]->apply( next_, current_, tmp_, M_ );
	
  // Exponentiate and normalize
  expAndNormalize( current_, next_, 1.0, relax );
}

void DenseCRF::currentMap( short * result )
{
  // Find the map
  for ( int i=0; i<N_; i++ )
  {
    const float * p = current_ + i*M_;
    // Find the max and subtract it so that the exp doesn't explode
    float mx = p[0];
    int imx = 0;
    for( int j=1; j<M_; j++ )
      if( mx < p[j] ){
	mx = p[j];
	imx = j;
      }
    result[i] = imx;
  }
}

////////////////////////////////////////////////
/////////////  GD and coocurence  //////////////
////////////////////////////////////////////////

float *DenseCRF2DGD::runInference(int n_iterations, float relax)
{
  startInference();

  double runtime = 0.0;

  CPrecisionTimer timer;
  for (int it = 0; it < n_iterations; ++it)
  {
    curr_iter_ = it;
    timer.Start();
    stepInference(relax);
    runtime += timer.Stop();

    runtimes_.push_back(runtime);
    kls_.push_back(computeKLDivergence());
  }
  return current_;
}

void DenseCRF2DGD::startInference()
{
  // initializing only based on unary potentials
  float *nat = nat_.data();
  float *nat_up = next_;
  for (int i = 0; i < N_; ++i)
  {
    for (int l = 0; l < M_; ++l)
    {
      nat[i*M_+l] = -unary_[i*M_+l];
      nat_up[i*M_+l] = -unary_[i*M_+l];
    }
  }
  expAndNormalize();  

  // initializing coocurrence potentials
  startInferenceCooc();
  expAndNormalizeCooc();
}

void DenseCRF2DGD::startInferenceCooc()
{
  if (!hasCooc())
    return;

  using namespace std;
  const float *q = current_;

  // writing these first
  float *nat_c = cooc_nat_.data();
  float *nat_up_c = cooc_next_.data();

  // uniform initialization
  // background is always there
  nat_c[0] = -1000;
  nat_c[1] = 0;
  for (int l = 1; l < M_; ++l)
  {
    nat_c[l*2] = log(0.5); 
    nat_c[l*2+1] = log(0.5); 
  }
}

void DenseCRF2DGD::stepInference(float relax)
{
  // this is current posterior
  const float *q = current_;
  // this are updated natural parameters
  float *nat_up = next_;

  // computing expectations wrt q-s
  for( int i=0; i<N_*M_; i++ )
    nat_up[i] = -unary_[i] - additional_unary_[i];
  for (size_t i = 0; i < pairwise_.size(); i++)
    pairwise_[i]->apply(nat_up, q, tmp_, M_);

  // computing expectations for cooc potentials
  stepInferenceCooc();


  // it is too different for adhoc smoothing
  if (inference_type_ == INFERENCE_ADHOC)
  {
    stepInferenceAdhoc();
    return;
  }

  if (hasCooc())
  {
    float *nat_c = cooc_nat_.data();
    const float *nat_up_c = cooc_next_.data();
    // making a step for cooc
    for (int l = 1; l < M_; ++l)
      for (int b = 0; b < 2; ++b)
	nat_c[l*2+b] = cooc_step_ * nat_up_c[l*2+b] + (1.0 - cooc_step_) * nat_c[l*2+b];
    cooc_step_ *= 1.15;
    cooc_step_ = std::min(cooc_step_, 1.0);
  }
  
  // making a step for pixels
  float *nat = nat_.data();
  if (inference_type_ == INFERENCE_NAT)
  {
    for (int i = 0; i < N_; ++i)
      for (int l = 0; l < M_; ++l)
      {
	int idx = i*M_+l;
	nat[idx] = step_ * nat_up[idx] + (1.0 - step_) * nat[idx];
      }
  }
  else if (inference_type_ == INFERENCE_NAT_MOMENTUM)
  {
    for (int i = 0; i < N_; ++i)
      for (int l = 0; l < M_; ++l)
      {
	int idx = i*M_+l;
	velocity_[idx] = gamma_ * velocity_[idx] + (1.0 - gamma_) * step_ * (nat[idx] - nat_up[idx]);
	nat[idx] = nat[idx] - velocity_[idx];
      }
  }
  else if (inference_type_ == INFERENCE_NAT_ADAM)
  {
    for (int i = 0; i < N_; ++i)
      for (int l = 0; l < M_; ++l)
      {
	int idx = i*M_+l;
	double grad = nat[idx] - nat_up[idx];
	adam_m1_[idx] = adam_b1_ * adam_m1_[idx] + (1.0-adam_b1_) * step_ * grad;
	adam_m2_[idx] = adam_b2_ * adam_m2_[idx] + (1.0-adam_b2_) * step_ * grad * grad;
	nat[idx] = nat[idx] - adam_m1_[idx] / (sqrt(adam_m2_[idx]) + adam_eps_);
      }
  }
  else if (inference_type_ == INFERENCE_NAT_ADAM_ORIG)
  {
    double t = curr_iter_+1;
    double lambda = 1.0-1e-8;
    double b1_t = adam_b1_ * pow(lambda, t);

    double m1_fix = (1.0 - pow(adam_b1_, t));
    double m2_fix = (1.0 - pow(adam_b2_, t));

    for (int i = 0; i < N_; ++i)
      for (int l = 0; l < M_; ++l)
      {
    	int idx = i*M_+l;
    	double grad = nat[idx] - nat_up[idx];
    	adam_m1_[idx] = b1_t * adam_m1_[idx] + (1.0-b1_t) * grad;
    	adam_m2_[idx] = adam_b2_ * adam_m2_[idx] + (1.0-adam_b2_) * grad * grad;
    	nat[idx] = nat[idx] - step_ * (adam_m1_[idx] / m1_fix) / (sqrt(adam_m2_[idx] / m2_fix) + adam_eps_);
      }
  }
  else
    throw std::runtime_error("inference type not implemented");
	
  // Exponentiate and normalize
  expAndNormalize();
  expAndNormalizeCooc();
}

void DenseCRF2DGD::stepInferenceAdhoc()
{
  float *q = current_;
  const float *nat_up = next_;

  for (int i = 0; i < N_; ++i)
  {
    float max_nat_up = nat_up[i*M_];
    for (int l = 1; l < M_; ++l)
      max_nat_up = std::max(nat_up[i*M_+l], max_nat_up);

    std::vector<float> q_exp(M_);    
    double normalizer = 0.0;
    for (int l = 0; l < M_; ++l)
    {
      q_exp[l] = fast_exp(nat_up[i*M_+l] - max_nat_up);
      normalizer += q_exp[l];
    }
    // smoothing
    for (int l = 0; l < M_; ++l)
      q[i*M_+l] = step_ * (q_exp[l] / normalizer) + (1.0 - step_) * q[i*M_+l];
  }

  // cooc
  if (!hasCooc())
    return;

  // current estimate of the natural parameters
  const float *nat_up_c = cooc_next_.data();
  // save results here for the normalized probabilities
  float *q_c = cooc_q_.data();
  
  for (int l = 1; l < M_; ++l)
  {
    float max_nat_up_c = std::max(nat_up_c[l*2], nat_up_c[l*2+1]);
    double normalizer = 0.0;
    std::vector<double> q_exp(2, 0);
    for (int b = 0; b < 2; ++b)
    {
      q_exp[b] = fast_exp(nat_up_c[l*2+b] - max_nat_up_c);
      normalizer += q_exp[b];
    }
    for (int b = 0; b < 2; ++b)
      q_c[l*2+b] = cooc_step_ * (q_exp[b] / normalizer) + (1.0 - cooc_step_) * q_c[l*2+b];
  }
}

void DenseCRF2DGD::stepInferenceCooc()
{
  if (!hasCooc())
    return;

  const float *q = current_;
  float *nat_up = next_;

  // current posterior probabilites
  float *q_c = cooc_q_.data();
  // update of the natural parameters (cooc)
  float *nat_up_c = cooc_next_.data();
  // current estimate of natural parameters (cooc)
  float *nat_c = cooc_nat_.data();

  std::vector<double> sum_q(2*M_, 0.0);

  for (int l = 1; l < M_; ++l)
  {
    for (int i = 0; i < N_; ++i)
    {
      sum_q[l*2] += (1.0 - q[i*M_+l]);
      sum_q[l*2+1] += q[i*M_+l];
    }

    sum_q[l*2] = std::max(sum_q[l*2], 1e-4);
    sum_q[l*2+1] = std::max(sum_q[l*2+1], 1e-4);
  }

  // Y_l = 0.
  // penalizing the absence of class l by all those pixels which belong to l
  for (int l = 1; l < M_; ++l)
    nat_up_c[l*2] = -cooc_factor_ * sum_q[l*2+1];

  // Y_l = 1. sum_{l' != l} C_{l,l'} * Q(Y_{l'} = 1)
  for (int l_a = 1; l_a < M_; ++l_a)
    for (int l_b = 1; l_b < M_; ++l_b)
    {
      double psum = 0.0;
      for (int l_b = 1; l_b < M_; ++l_b)
	if (l_a != l_b)
	  psum += cooc_logp_[l_a*M_+l_b] * q_c[l_b*2+1];
      // penalizing the presence of each based on the probability
      // of seeing the other labels together with this one
      nat_up_c[l_a*2+1] = psum;    
    }

  // adding coocurence terms to the expectations of X
  // TODO: why is it using the old q_с?
  if (curr_iter_ > 1)
  {
    // penalizing pixels with those classes not present
    for (int i = 0; i < N_; ++i)
      for (int l = 1; l < M_; ++l)
	nat_up[i*M_+l] -= cooc_factor_ * q_c[l*2];
  }
}



void DenseCRF2DGD::expAndNormalize(float *, const float *, float , float )
{
  float *q = current_;
  //float *V = new float[ N_+10 ];
  // over all the variables
  for (int i = 0; i < N_; i++)
  {
    const float *nat_i = nat_.data() + i * M_;
    
    // get the exp max
    float max_nat = nat_i[0];
    for (int l = 1; l < M_; ++l)
      if (max_nat < nat_i[l])
	max_nat = nat_i[l];

    // taking exp
    std::vector<float> q_exp(M_);
    float normalizer = 0.0;
    for (int l = 0; l < M_; ++l)
    {
      q_exp[l] = fast_exp(nat_i[l] - max_nat);
      normalizer += q_exp[l];
    }

    // saving normalized probabilites
    for (int l = 0; l < M_; ++l)
      q[i*M_ + l] = q_exp[l] / normalizer;
  }
}

void DenseCRF2DGD::expAndNormalizeCooc()
{
  if (!hasCooc())
    return;
  // current estimate of the natural parameters
  const float *nat_c = cooc_nat_.data();
  // save results here for the normalized probabilities
  float *q_c = cooc_q_.data();
  
  for (int l = 0; l < M_; ++l)
  {
    float max_nat_c = std::max(nat_c[l*2], nat_c[l*2+1]);

    double normalizer = 0.0;
    std::vector<double> q_exp(2);
    for (int b = 0; b < 2; ++b)
    {
      q_exp[b] = fast_exp(nat_c[l*2+b] - max_nat_c);
      normalizer += q_exp[b];
    }
    
    for (int b = 0; b < 2; ++b)
      q_c[l*2+b] = q_exp[b] / normalizer;
  }
}

void DenseCRF2DGD::setCooc(const std::vector<float> &unary,
			   const std::vector<float> &pairwise,
			   const std::vector<float> &counts,			   
			   float factor,
			   float step)
{
  cooc_unary_ = unary;
  cooc_pairwise_ = pairwise;
  cooc_counts_ = counts;
  cooc_factor_ = factor;
  cooc_step_ = step;

  // hidden state of the potentials (class probabilities on image-level)
  cooc_next_.resize(2*M_, 0.0);
  cooc_q_.resize(2*M_, 0.0);
  cooc_nat_.resize(2*M_, 0.0);
  cooc_logp_.resize(M_*M_, 0.0);

  // pre-computing cooccurrence potentials
  for (int l_a = 1; l_a < M_; ++l_a)
    for (int l_b = 1; l_b < M_; ++l_b)
    {
      double p_a = cooc_unary_[l_a];      
      double p_b = cooc_unary_[l_b];
      double p_ab = cooc_pairwise_[l_a*M_+l_b];
      double p = 1.0 - (1.0 - p_ab / p_b) * (1.0 - p_ab / p_a);
      p = std::max(std::min(p, 1.0), 1e-6);
      cooc_logp_[l_a*M_+l_b] = 0.005 * N_ * log(p);
    }
}

double DenseCRF2DGD::computeKLDivergence() const
{
  double kl = 0.0;

  const double eps_q = 1e-10;

  const float *q = current_;

  // entropy
  for (int i = 0; i < N_; ++i)
    for (int l = 0; l < M_; ++l)
      kl += q[i*M_+l] * log(std::max<double>(q[i*M_+l], eps_q));

  // unary 
  for (int i = 0; i < N_; ++i)
    for (int l = 0; l < M_; ++l)
      kl += q[i*M_+l] * unary_[i*M_+l];

  // pairwise
  std::vector<float> tmp(N_*M_, 0);
  std::vector<float> nat_pairwise(N_*M_, 0);

  for (int p = 0; p < pairwise_.size(); ++p)
    pairwise_[p]->apply(nat_pairwise.data(), q, tmp.data(), M_);
  for (int i = 0; i < N_; ++i)
    for (int l = 0; l < M_; ++l)
      kl += - q[i*M_+l] * nat_pairwise[i*M_+l];

  if (hasCooc())
  {
    const float *q_c = cooc_q_.data();

    // entropy
    for (int l = 0; l < M_; ++l)
      for (int b = 0; b < 2; ++b)
	kl += q_c[l*2+b] * log(std::max<double>(q[l*2+b], eps_q));

    // log p(X_i = l, Z_i = 0) Q(Z_i = 0)
    for (int i = 0; i < N_; ++i)
      for (int l = 0; l < M_; ++l)
	kl += q[i*M_+l] * q_c[l*2] * cooc_factor_;

    // log p(Z_a = 1, Z_b = 1) Q(Z_a = 1) Q(Z_a = 1)
    for (int l_a = 1; l_a < M_; ++l_a)
      for (int l_b = l_a+1; l_b < M_; ++l_b)
	kl += - q_c[l_a*2+1] * q_c[l_b*2+1] * cooc_logp_[l_a*M_+l_b];
  }

  return kl;
}
