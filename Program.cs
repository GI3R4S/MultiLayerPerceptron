using System;
using System.Collections.Generic;
using System.IO;
using System.Xml.Serialization;
using static System.Console;
using static System.Math;
namespace MLP
{
    public partial class MLP
    {
        #region Enums
        public enum Variant
        {
            transformation,
            aproximation
        }
        public enum Bias
        {
            biasOff,
            biasOn
        }
        #endregion

        #region Parameters
        static Variant variant = Variant.aproximation;
        static Bias bias = Bias.biasOn;
        static int hiddenLayerCount = 4;
        static int epochsCount = 1000;
        static double executionsCount = 5;
        static double learningFactor = 0.5;
        static double momentumFactor = 0.5;
        static double sigmoidSteepnessFactor = 1;
        static int trainingSetNumber = 2;  //Applies to aproximation variant
        #endregion

        public static void Main(String[] args)
        {
            Random gen = new Random();
            if (variant == Variant.transformation)
            {

                for (int counter = 1; counter <= executionsCount; counter++)
                {
                    string fileName = variant.ToString() + "_" + bias.ToString() + "_Execution" + counter.ToString() + "EpochsDiffrences.xml";
                    StreamWriter sw = new StreamWriter(fileName);
                    XmlSerializer xs = new XmlSerializer(typeof(List<double>));
                    List<double> EpochsMSEs = new List<double>();
                    double[][] testSamples = LoadTrainingDataFromFileTransformation();
                    double[][] finalInputOutput = null;
                    List<double[]> trainingSet = new List<double[]>();
                    RefillTrainingSet(trainingSet, testSamples);


                    Neuron [] hiddenLayer = null;
                    Neuron [] outputLayer = null;
                    InitalizeLayers(ref hiddenLayer, ref outputLayer);
                    for(int i = 1; i <= epochsCount; i++)
                    {
                        double EpochMSE = 0;
                        double IterationError = 0;
                        EpochMSE = 0;
                        for (int j = trainingSet.Count; j > 0; j--)
                        {
                            IterationError = 0;
                            int randomIndex = gen.Next(j);
                            double[] inputs1 = trainingSet[randomIndex];
                            double[] inputs2 = new double[hiddenLayerCount];
                            foreach (Neuron n in hiddenLayer)
                            {
                                n.Inputs = inputs1;
                            }
                            for (int k = 0; k < hiddenLayer.Length; k++)
                            {
                                inputs2[k] = hiddenLayer[k].Output();
                            }
                            foreach (Neuron n in outputLayer)
                            {
                                n.Inputs = inputs2;
                            }

                            double[] outputsErrors = new double[4];
                            for (int k = 0; k < outputLayer.Length; k++)
                            {
                                outputsErrors[k] = (inputs1[k] - outputLayer[k].Output());
                                IterationError += Pow(outputsErrors[k], 2);

                            }
                            for (int k = 0; k < outputLayer.Length; k++)
                            {
                                outputLayer[k].Error = Sigm.FunctionDerivative(outputLayer[k].Output()) * (outputsErrors[k]);
                                outputLayer[k].UpdateWeights();
                            }

                            for (int k = 0; k < hiddenLayer.Length; k++)
                            {
                                double value = 0;
                                for (int l = 0; l < hiddenLayer[k].Weights.Length; l++)
                                {
                                    value += Sigm.FunctionDerivative(hiddenLayer[k].Output()) * outputLayer[l].Error * outputLayer[l].Weights[k];
                                }
                                hiddenLayer[k].Error = value;
                            }
                            for (int k = 0; k < hiddenLayer.Length; k++)
                            {
                                hiddenLayer[k].UpdateWeights();
                            }
                            trainingSet.RemoveAt(randomIndex);
                            EpochMSE += IterationError;
                            if (i == epochsCount && j == 1)
                            {
                                finalInputOutput = new double[2][];
                                XmlSerializer xs1 = new XmlSerializer(typeof(double[][]));
                                finalInputOutput[0] = inputs1;
                                finalInputOutput[1] = new double[] { outputLayer[0].Output(), outputLayer[1].Output(), outputLayer[2].Output(), outputLayer[3].Output() };
                                using (StreamWriter sw1 = new StreamWriter(variant.ToString() + "_" + bias.ToString() + "_Execution" + counter.ToString() + "FinalInputOutput.xml"))
                                {
                                    xs1.Serialize(sw1, finalInputOutput);
                                }
                            }
                        }
                        EpochMSE /= 4;
                        RefillTrainingSet(trainingSet, testSamples);
                        if (i % 20 == 1)
                            EpochsMSEs.Add(EpochMSE);
                    }

                    xs.Serialize(sw, EpochsMSEs);
                    PrintEpochResult(finalInputOutput);

                }
            }


            if (variant == Variant.aproximation)
            {
                for (int counter = 1; counter <= executionsCount; counter++)
                {
                    StreamWriter sw = new StreamWriter(variant.ToString() + "_" + bias.ToString() + "_Execution" + counter.ToString() + "EpochsDiffrences.xml");
                    XmlSerializer xs = new XmlSerializer(typeof(List<ApproximationData>));
                    List<ApproximationData> toSerialize = new List<ApproximationData>();
                    List<double> trainingDataInputs = new List<double>();
                    List<double> trainingDataOutputs = new List<double>();
                    List<double> testingDataInputs = new List<double>();
                    List<double> testingDataOutputs = new List<double>();
                    LoadTrainingDataFromFileAproximation(trainingDataInputs, trainingDataOutputs, testingDataInputs, testingDataOutputs);


                    Neuron[] hiddenLayer = new Neuron[hiddenLayerCount];
                    Neuron[] outputLayer = new Neuron[1];
                    for (int i = 0; i < hiddenLayer.Length; i++)
                    {
                        hiddenLayer[i] = new Neuron(1, 1);
                        hiddenLayer[i].RandomizeValues();
                    }
                    outputLayer[0] = new Neuron(hiddenLayerCount, 2);
                    outputLayer[0].RandomizeValues();
                    double TrainingMSE = 0;


                    for(int i = 1; i <= epochsCount; i++)
                    { 
                        List<int> numbers = GetNumbers(trainingDataInputs.Count);
                        List<double> finalOutput = new List<double>();
                        TrainingMSE = 0;
                        for (int j =  0 ; j < trainingDataInputs.Count; j++)
                        {
                            int randomIndex = gen.Next(numbers.Count);
                            numbers.RemoveAt(randomIndex);
                            double[] hiddenLayerInputs = new double[] { trainingDataInputs[randomIndex] };
                            double[] outputLayerInputs = new double[hiddenLayerCount];


                            foreach (Neuron n in hiddenLayer)
                            {
                                n.Inputs = hiddenLayerInputs;
                            }
                            for (int k = 0; k < hiddenLayer.Length; k++)
                            {
                                outputLayerInputs[k] = hiddenLayer[k].Output();
                            }
                            outputLayer[0].Inputs = outputLayerInputs;


                            double diffrence = 0;
                            diffrence = trainingDataOutputs[randomIndex] - outputLayer[0].Output();
                            TrainingMSE += Pow(diffrence, 2);

                            outputLayer[0].Error = Linear.FunctionDerivative(outputLayer[0].Output()) * diffrence;
                            for (int k = 0; k < hiddenLayer.Length; k++)
                            {
                                hiddenLayer[k].Error = Sigm.FunctionDerivative(hiddenLayer[k].Output()) * outputLayer[0].Error * outputLayer[0].Weights[k];
                                hiddenLayer[k].UpdateWeights();
                            }
                            outputLayer[0].UpdateWeights();
                        }


                        TrainingMSE /= trainingDataInputs.Count;
                        double TestingMSE = 0;


                        for (int j = 0; j < testingDataInputs.Count; j++)
                        {
                            double[] hiddenLayerInputs = new double[] { testingDataInputs[j] };
                            double[] outputLayerInputs = new double[hiddenLayerCount];


                            foreach (Neuron n in hiddenLayer)
                            {
                                n.Inputs = hiddenLayerInputs;
                            }
                            for (int k = 0; k < hiddenLayer.Length; k++)
                            {
                                outputLayerInputs[k] = hiddenLayer[k].Output();
                            }
                            outputLayer[0].Inputs = outputLayerInputs;

                            TestingMSE += Pow(testingDataOutputs[j] - outputLayer[0].Output(), 2);
                            if (i == epochsCount)
                            {
                                finalOutput.Add(outputLayer[0].Output());
                            }
                        }
                        if (i == epochsCount)
                        {
                            XmlSerializer xs1 = new XmlSerializer(typeof(List<double>));
                            using (StreamWriter sw1 = new StreamWriter(variant.ToString() + "_" + bias.ToString() + "_Execution" + counter.ToString() + "FinalOuput.xml"))
                            {
                                xs1.Serialize(sw1, finalOutput);
                            }
                        }
                        TestingMSE /= testingDataInputs.Count;
                        ApproximationData approximationData;
                        approximationData.MSETrening = TrainingMSE;
                        approximationData.MSETest = TestingMSE;
                        if (i % 20 == 1)
                            toSerialize.Add(approximationData);
                    }

                    xs.Serialize(sw, toSerialize);
                }
            }
        }
        public static double [][] LoadTrainingDataFromFileTransformation()
        {
            StreamReader sr = new StreamReader("../../resources/transformation.txt");
            double[][] testSamples = new double[4][];
            string line;
            line = sr.ReadToEnd();
            line = line.Replace(" ", "");
            line = line.Replace(Environment.NewLine, "");
            line = line.Replace("\n", "");
            testSamples[0] = new double[4];
            testSamples[1] = new double[4];
            testSamples[2] = new double[4];
            testSamples[3] = new double[4];
            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    testSamples[i][j] = (char.GetNumericValue(line[i * 4 + j]));
                }
            }
            return testSamples;
        }

        public static void InitalizeLayers(ref Neuron [] hiddenLayer,ref Neuron [] outputLayer)
        {
            hiddenLayer = new Neuron[hiddenLayerCount];
            outputLayer = new Neuron[4];
            for (int i = 0; i < hiddenLayer.Length; i++)
            {
                hiddenLayer[i] = new Neuron(4, 1);
                hiddenLayer[i].RandomizeValues();

            }
            for (int i = 0; i < outputLayer.Length; i++)
            {
                outputLayer[i] = new Neuron(hiddenLayerCount, 1);
                outputLayer[i].RandomizeValues();
            }
        }

        public static void RefillTrainingSet(List<double[]> trainingSet, double [][] testSamples)
        {
            trainingSet.Add(testSamples[0]);
            trainingSet.Add(testSamples[1]);
            trainingSet.Add(testSamples[2]);
            trainingSet.Add(testSamples[3]);
        }

        public static void PrintEpochResult(double [][] finalInputOutput)
        {
            WriteLine("Final output: \n ");
            for (int j = 0; j < finalInputOutput[0].GetLength(0); j++)
            {
                WriteLine(finalInputOutput[0][j] + " - " + finalInputOutput[1][j]);
            }
            WriteLine("Press any button to continue..");
            ReadKey();
        }

        public static void LoadTrainingDataFromFileAproximation(List<double> trainingDataInputs, List<double> trainingDataOutput, List<double> testingDataInputs, List<double> testingDataOutputs)
        {
            StreamReader streamReader = new StreamReader("../../resources/approximation_train_1.txt");
            string string1 = streamReader.ReadToEnd();
            streamReader = new StreamReader("../../resources/approximation_train_2.txt");
            string string2 = streamReader.ReadToEnd();
            string1 = string1.Replace('\n', ' ');
            string2 = string2.Replace('\n', ' ');
            string1 = string1.Replace('.', ',');
            string2 = string2.Replace('.', ',');
            string[] traningSet1 = string1.Split(' ');
            string[] trainingSet2 = string2.Split(' ');
            if (trainingSetNumber == 1)
            {
                for (int i = 0; i < traningSet1.Length - 1; i++)
                {
                    if (i % 2 == 0)
                    {
                        trainingDataInputs.Add(double.Parse(traningSet1[i]));
                    }
                    else
                    {
                        trainingDataOutput.Add(double.Parse(traningSet1[i]));
                    }
                }
            }
            if (trainingSetNumber == 2)
            {
                for (int i = 0; i < trainingSet2.Length - 1; i++)
                {
                    if (i % 2 == 0)
                    {
                        trainingDataInputs.Add(double.Parse(trainingSet2[i]));
                    }
                    else
                    {
                        trainingDataOutput.Add(double.Parse(trainingSet2[i]));
                    }
                }
            }
            streamReader = new StreamReader("../../resources/approximation_test.txt");
            string string3 = streamReader.ReadToEnd();
            string3 = string3.Replace('\n', ' ');
            string3 = string3.Replace(Environment.NewLine, " ");
            string3 = string3.Replace('.', ',');
            string[] testingSet = string3.Split(' ');


            for (int i = 0; i < testingSet.Length - 1; i++)
            {
                if (i % 2 == 0)
                {
                    testingDataInputs.Add(double.Parse(testingSet[i]));
                }
                else
                {
                    testingDataOutputs.Add(double.Parse(testingSet[i]));
                }
            }

        }

        public static List<int> GetNumbers(int range)
        {
            List<int> numbers = new List<int>();
            for (int i = 0; i < range; i++)
            {
                numbers.Add(i);
            }
            return numbers;
        }
    }
}