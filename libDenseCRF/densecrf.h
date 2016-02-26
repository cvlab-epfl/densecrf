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

#pragma once

#include <vector>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <cstdarg>

inline std::string string_format(const std::string fmt, ...)
{
  int size = 512;
  std::string str;
  va_list ap;
  while (1) {
    str.resize(size);
    va_start(ap, fmt);
    int n = vsnprintf((char *)str.c_str(), size, fmt.c_str(), ap);
    va_end(ap);
    if (n > -1 && n < size) {
      str.resize(n);
      return str;
              
    }
    if (n > -1)
      size = n + 1;
    else
      size *= 2;
        
  }
  return str;
}


class PairwisePotential
{
public:
  virtual ~PairwisePotential();
  virtual void apply(float * out_values, const float * in_values, float * tmp, int value_size) const = 0;
};

class SemiMetricFunction
{
public:
  virtual ~SemiMetricFunction();
  // For two probabilities apply
  // the semi metric transform: v_i = sum_j mu_ij u_j
  virtual void apply(float * out_values, const float * in_values, int value_size) const = 0;
};

class DenseCRF
{
protected:
  friend class BipartiteDenseCRF;
	
  // Number of variables and labels
  int N_, M_;
  float *unary_, *additional_unary_, *current_, *next_, *tmp_;
  int curr_iter_;
	
  // Store all pairwise potentials
  std::vector<PairwisePotential*> pairwise_;
	
  // Run inference and return the pointer to the result
  virtual float* runInference(int n_iterations, float relax);
	
  // Auxillary functions
  virtual void expAndNormalize(float* out, const float* in, float scale = 1.0, float relax = 1.0);
	
  // Don't copy this object, bad stuff will happen
  DenseCRF(DenseCRF & o) {}
public:
  // Create a dense CRF model of size N with M labels
  DenseCRF(int N, int M);
  virtual ~DenseCRF();
  // Add  a pairwise potential defined over some feature space
  // The potential will have the form:    w*exp(-0.5*|f_i - f_j|^2)
  // The kernel shape should be captured by transforming the
  // features before passing them into this function
  void addPairwiseEnergy(const float * features, int D, float w = 1.0f, const SemiMetricFunction * function = NULL);
	
  // Add your own favorite pairwise potential (ownwership will be transfered to this class)
  void addPairwiseEnergy(PairwisePotential* potential);
	
  // Set the unary potential for all variables and labels (memory order is [x0l0 x0l1 x0l2 .. x1l0 x1l1 ...])
  void setUnaryEnergy(const float * unary);
	
  // Set the unary potential for a specific variable
  void setUnaryEnergyForN(int n, const float * unary);
	
  // Run inference and return the probabilities
  void inference(int n_iterations, float* result, float relax = 1.0);
	
  // Run MAP inference and return the map for each pixel
  void map(int n_iterations, short int* result, float relax = 1.0);
	
  // Step by step inference
  virtual void startInference();
  virtual void stepInference(float relax = 1.0);
  void currentMap(short * result);
	
public: /* Debugging functions */
  // Compute the unary energy of an assignment
  void unaryEnergy(const short * ass, float * result);
	
  // Compute the pairwise energy of an assignment (half of each pairwise potential is added to each of it's endpoints)
  void pairwiseEnergy(const short * ass, float * result, int term = -1);
};

class DenseCRF2D : public DenseCRF
{
protected:
  // Width, height of the 2d grid
  int W_, H_;

public:
  // Create a 2d dense CRF model of size W x H with M labels
  DenseCRF2D(int W, int H, int M);
  virtual ~DenseCRF2D();
  // Add a Gaussian pairwise potential with standard deviation sx and sy
  void addPairwiseGaussian(float sx, float sy, float w,
			   const SemiMetricFunction * function = NULL);
	
  // Add a Bilateral pairwise potential with spacial
  // standard deviations sx, sy and color standard deviations sr,sg,sb
  void addPairwiseBilateral(float sx, float sy, float sr, float sg, float sb, const unsigned char * im, float w, const SemiMetricFunction * function = NULL);
	
  // Set the unary potential for a specific variable
  void setUnaryEnergyForXY(int x, int y, const float * unary);
  //using DenseCRF::setUnaryEnergy;
};

// A dense CRF in a bipartite graph
class BipartiteDenseCRF {
protected:
  // Two dense CRF's that are connected by a set of completely connected edges (in a bipartite graph)
  DenseCRF* dense_crfs_[2];
	
  // Number of variables and labels
  int N_[2], M_;
	
  // All bipartite pairwise potentials (all others are stored in each dense_crfs respectively)
  std::vector<PairwisePotential*> pairwise_[2];
	
  // Don't copy this object, bad stuff will happen
  BipartiteDenseCRF(BipartiteDenseCRF & o){}
	
  // Run inference and return the pointer to the result
  void runInference(int n_iterations, float ** prob, float relax);
public:
  // Create a dense CRF model of size N with M labels
  BipartiteDenseCRF(int N1, int N2, int M);
  ~BipartiteDenseCRF();
	
  // Add  a pairwise potential defined over some feature space
  // The potential will have the form:    w*exp(-0.5*|f_i - f_j|^2)
  // The kernel shape should be captured by transforming the
  // features before passing them into this function
  void addPairwiseEnergy(const float * features1, const float * features2, int D, float w = 1.0f, const SemiMetricFunction * function = NULL);
	
  // Add your own favorite pairwise potential (ownwership will be transfered to this class)
  void addPairwiseEnergy(PairwisePotential* potential12,
			 PairwisePotential* potential21);
	
  // Run inference and return the probabilities
  void inference(int n_iterations, float* result1, float * result2, float relax = 1);
	
  // Run MAP inference and return the map for each pixel
  void map(int n_iterations, short int* result1, short int* result2, float relax = 1);
	
  // Access the two CRF's directly
  DenseCRF& getCRF(int i);
  const DenseCRF& getCRF(int i) const;
    
  // Step by step inference
  void startInference();
  void stepInference(float relax = 1.0);
  void currentMap(short * result);
};

// This function defines a simplified interface to the permutohedral lattice
// We assume a filter standard deviation of 1
class Permutohedral;
class Filter{
protected:
  int n1_, o1_, n2_, o2_;
  Permutohedral * permutohedral_;
  // Don't copy
  Filter( const Filter& filter ){}
public:
  // Use different source and target features
  Filter( const float * source_features, int N_source, const float * target_features, int N_target, int feature_dim );
  // Use the same source and target features
  Filter( const float * features, int N, int feature_dim );
  //
  ~Filter();
  // Filter a bunch of values
  void filter( const float * source, float * target, int value_size );
};


class DenseCRF2DGD : public DenseCRF2D
{
public:
  enum InferenceType
  {
    INFERENCE_NAT = 0,
    INFERENCE_NAT_MOMENTUM = 1,
    INFERENCE_NAT_ADAM = 2,
    INFERENCE_NAT_ADAM_ORIG = 3,
    INFERENCE_ADHOC = 4
  };
  
  DenseCRF2DGD(int W, int H, int M,
	       InferenceType inference_type,
	       double step,
	       double gamma,
               double adam_b1=0.9, double adam_b2=0.99, double adam_eps=1e-8):
    DenseCRF2D(W, H, M),
    inference_type_(inference_type),
    step_(step),
    nat_(W*H*M, 0.0),
    // momentum
    gamma_(gamma),
    velocity_(W*H*M, 0.0),
    adam_b1_(adam_b1), adam_b2_(adam_b2), adam_eps_(adam_eps),
    adam_m1_(W*H*M, 0.0), adam_m2_(W*H*M, 0.0)
  {}

  // TODO: re-define runInference() and map() ?
  float *runInference(int n_iterations, float relax);
  void startInference();
  void stepInference(float relax);
  void stepInferenceAdhoc();
  void expAndNormalize(float* q = NULL, const float *nat = NULL, float scale = 1.0, float relax = 1.0);


  // set co-occurence potentials
  void setCooc(const std::vector<float> &unary,
	       const std::vector<float> &pairwise,
	       const std::vector<float> &counts,
	       float factor, float step);
  
  void startInferenceCooc();
  void stepInferenceCooc();
  void expAndNormalizeCooc();

  double computeKLDivergence() const;

  inline bool hasCooc() const
  {
    return !cooc_unary_.empty();
  }

  inline const std::vector<double> runtimes() const { return runtimes_; }
  inline const std::vector<double> kls() const { return kls_; }

protected:
  InferenceType inference_type_;
  // gd step size for pixels
  double step_;
  // current estimate of the natural parameters
  std::vector<float> nat_;

  // momentum parameters
  double gamma_;
  std::vector<double> velocity_;

  // adam parameters
  double adam_b1_, adam_b2_, adam_eps_;
  std::vector<double> adam_m1_, adam_m2_;

  // co-occurence parameters
  double cooc_step_;
  std::vector<float> cooc_unary_, cooc_pairwise_, cooc_counts_;
  std::vector<double> cooc_logp_;
  float cooc_factor_;
  std::vector<float> cooc_q_, cooc_next_, cooc_nat_;

  // momentum for cooc
  std::vector<double> cooc_velocity_;

  // running times
  std::vector<double> runtimes_;
  std::vector<double> kls_;
};

