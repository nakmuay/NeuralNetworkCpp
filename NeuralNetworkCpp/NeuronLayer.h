#pragma once

#include "ActivationFunction.h"

#include <vector>

using namespace std;

namespace NeuralNetwork
{

  class NeuronLayer
  {
  protected:
    const int m_size;

  private:
    const IDoubleEvaluatable& m_activationFunction;

  public:
    NeuronLayer(int size, IDoubleEvaluatable& activationFunction)
    : m_size(size),
      m_activationFunction(activationFunction)
    {
    }

    int getSize()
    {
      return m_size;
    }

    const IDoubleEvaluatable& getActivationFunction()
    {
        return m_activationFunction;
    }

    void Run(vector<double>& input, vector<double>* &output)
    {
      RunCore(input, output);
    }

    virtual void RunCore(vector<double>& input, vector<double>* &output)
    {
      int size = input.size();
      output = new vector<double>(size);

      for (int i = 0; i < size; i++)
      {
        (*output)[i] = m_activationFunction.Evaluate(input[i]);
      }
    }

    void CalculateDeltas(vector<double>& input, vector<double>& backPropDeltas, vector<double>* &thisLayerDeltas)
    {
      thisLayerDeltas = new vector<double>(m_size);

      for (int i = 0; i < m_size; i++)
      {
        (*thisLayerDeltas)[i] = m_activationFunction.EvaluateDerivative(input[i]) * backPropDeltas[i];
      }
    }
  };
}
