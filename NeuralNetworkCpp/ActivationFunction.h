#pragma once

#include "stdafx.h"


namespace NeuralNetwork
{
  template <typename T>
  class IEvaluatable
  {
  public:
    virtual T Evaluate(T value)  const = 0;

    virtual T EvaluateDerivative(T value)  const = 0;
  };


  class IDoubleEvaluatable : public IEvaluatable<double>
  {
  };


  class None : public IDoubleEvaluatable
  {
  public:
    double Evaluate(double value);

    double EvaluateDerivative(double value);
  };


  class SigmoidActivationFunction : public IDoubleEvaluatable
  {
  public:
    double Evaluate(double value);

    double EvaluateDerivative(double value);
  };

  class TanhActivationFunction : public IDoubleEvaluatable
  {
  public:
    double Evaluate(double value);

    double EvaluateDerivative(double value);
  };


  class ModifiedTanhActivationFunction : public IDoubleEvaluatable
  {
  private: 
    double scale = 1.7159;
    double periodScale = 2.0 / 3.0;

  public:
    double Evaluate(double value);

    double EvaluateDerivative(double value);
  };

  class RectifiedLinearUnitFunction : public IDoubleEvaluatable
  {
  public:
    double Evaluate(double value);

    double EvaluateDerivative(double value);
  };


  class LeakyRectifiedLinearUnitFunction : public IDoubleEvaluatable
  {
  private: 
    double positiveHalfPlaneSlope = 1.0;
    double negativeHalfPlaneSlope = 0.01;

  public:
    double Evaluate(double value);

    double EvaluateDerivative(double value);
  };


  class LinearActivationFunction : public IDoubleEvaluatable
  {
  public:
    double Evaluate(double value);

    double EvaluateDerivative(double value);
  };
}
