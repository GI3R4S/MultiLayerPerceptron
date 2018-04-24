using System;
namespace MLP
{
    public partial class MLP
    {
        public class Sigm
        {
            public static double Function(double x)
            {
                return 1.0 / (1.0 + Math.Exp(-x * sigmoidSteepnessFactor));
            }

            public static double FunctionDerivative(double x)
            {
                return sigmoidSteepnessFactor * x * (1 - x);
            }
        }
    }
}