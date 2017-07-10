#include "stdafx.h"


#include "ActivationFunction.h"

using namespace NeuralNetwork;

// ---------------------------------------------------------------------------------------------------------------------
// IDoubleEvaluatable
// ---------------------------------------------------------------------------------------------------------------------

double None::Evaluate(double value)
{
  return value;
}

double None::EvaluateDerivative(double value)
{
  return 1.0;
}


// ---------------------------------------------------------------------------------------------------------------------
// SigmoidActivationFunction
// ---------------------------------------------------------------------------------------------------------------------

double SigmoidActivationFunction::Evaluate(double value)
{
  return 1.0 / (1.0 + exp(-value));
}

double SigmoidActivationFunction::EvaluateDerivative(double value)
{
  return Evaluate(value) * (1.0 - Evaluate(value));
}

// ---------------------------------------------------------------------------------------------------------------------
// TanhActivationFunction
// ---------------------------------------------------------------------------------------------------------------------

double TanhActivationFunction::Evaluate(double value)
{
  return tanh(value);
}

double TanhActivationFunction::EvaluateDerivative(double value)
{
  return 1.0 - pow(Evaluate(value), 2.0);
}

// ---------------------------------------------------------------------------------------------------------------------
// ModifiedTanhActivationFunction
// ---------------------------------------------------------------------------------------------------------------------

double ModifiedTanhActivationFunction::Evaluate(double value)
{
  return scale * tanh(periodScale * value);
}

// TODO [martin, 2017-04-04]: Fix the calcualtion of the derivative. This is currently incorrect!
double ModifiedTanhActivationFunction::EvaluateDerivative(double value)
{
  return scale * periodScale * (1.0 - pow(Evaluate(value), 2.0));
}

// ---------------------------------------------------------------------------------------------------------------------
// RectifiedLinearUnitFunction
// ---------------------------------------------------------------------------------------------------------------------

double RectifiedLinearUnitFunction::Evaluate(double value)
{
  return fmax(0.0, value);
}

double RectifiedLinearUnitFunction::EvaluateDerivative(double value)
{
  return value > 0.0 ? 1.0 : 0.0;
}

// ---------------------------------------------------------------------------------------------------------------------
// LeakyRectifiedLinearUnitFunction
// ---------------------------------------------------------------------------------------------------------------------

double LeakyRectifiedLinearUnitFunction::Evaluate(double value)
{
  return fmax(negativeHalfPlaneSlope * value, value);
}

double LeakyRectifiedLinearUnitFunction::EvaluateDerivative(double value)
{
  return value > 0.0 ? positiveHalfPlaneSlope : negativeHalfPlaneSlope;
}

// ---------------------------------------------------------------------------------------------------------------------
// LinearActivationFunction
// ---------------------------------------------------------------------------------------------------------------------

double LinearActivationFunction::Evaluate(double value)
{
  return value;
}

double LinearActivationFunction::EvaluateDerivative(double value)
{
  return 1.0;
}
