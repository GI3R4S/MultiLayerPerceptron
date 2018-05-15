using System;
namespace MLP
{
    public partial class MLP
    {

        #region Neuron
        public class Neuron
        {
            #region Definitions
            public double[] Inputs { get; set; }
            public double[] Weights { get; set; }
            public double[] PreviousChanges { get; set; }
            public double PreviousBiasChange { get; set; }
            public double Bias { get; set; }
            public static Random gen = new Random();
            public double Error;
            public int ActivationFunction;
            #endregion
            public Neuron(int inputCount, int newActivationFuntion)
            {
                Weights = new double[inputCount];
                PreviousChanges = new double[inputCount];
                ActivationFunction = newActivationFuntion;
            }

            public double Output()
            {
                double result = 0;
                for (int i = 0; i < Weights.Length; i++)
                {
                    result += Weights[i] * Inputs[i];
                }
                if (bias == MLP.Bias.biasOn)
                {
                    result += Bias;
                }
                if (ActivationFunction == 1)
                {
                    result = Sigm.Function(result);
                }
                if (ActivationFunction == 2)
                {
                    result = Linear.Function(result);
                }
                return result;
            }
            public void RandomizeValues()
            {
                Bias = (gen.NextDouble()- 0.5 ) * 2;
                for (int i = 0; i < Weights.Length; i++)
                {
                    PreviousChanges[i] = 0;
                    PreviousBiasChange = 0;
                    Weights[i] = (gen.NextDouble()- 0.5) * 2;
                }
            }

            public void UpdateWeights()
            {
                for (int i = 0; i < Weights.Length; i++)
                {
                    Weights[i] += Error * Inputs[i] * learningFactor + momentumFactor * PreviousChanges[i];
                    PreviousChanges[i] = Error * Inputs[i] * learningFactor + momentumFactor * PreviousChanges[i];
                }

                if (bias == MLP.Bias.biasOn)
                {
                    Bias += Error * learningFactor + momentumFactor * PreviousBiasChange;
                    PreviousBiasChange = Error * learningFactor + momentumFactor * PreviousBiasChange;
                }
            }
        }

#endregion

    }
}