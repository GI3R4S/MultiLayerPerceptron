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
        #region Parameters
        static int variant = 2;
        static int hiddenLayerCount = 4;
        static int erasCount = 1000;
        static double learningFactor = 0.5;
        static double momentumFactor = 0.5;
        static double sigmoidSteepnessFactor = 1;
        static bool bias = true;
        static double repetitions = 5;
        static int trainingSetNumber = 2;
        #endregion

        public static void Main(String[] args)
        {
            Random gen = new Random();
            #region Transformation 
            if (variant == 1)
            {

                for (int counter = 1; counter <= repetitions; counter++)
                {
                    #region Declarations
                    string fileName = "TRANS1NeuronBrakBiasu0100@" + counter.ToString() + ".xml";
                    StreamWriter sw = new StreamWriter(fileName);
                    XmlSerializer xs = new XmlSerializer(typeof(List<double>));
                    List<double> toSerialize = new List<double>();
                    string line;
                    StreamReader sr = new StreamReader("transformation.txt");
                    List<double[]> trainingSet = new List<double[]>();
                    int iteration = 1;
                    double MSE = 0;
                    double ErasMSE = 0;
                    #endregion

                    #region LOAD_SAMPLES
                    line = sr.ReadToEnd();
                    line = line.Replace(" ", "");
                    line = line.Replace(Environment.NewLine, "");
                    line = line.Replace("\n", "");
                    double[][] testSamples = new double[4][];
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
                    #endregion


                    trainingSet.Add(testSamples[0]);
                    trainingSet.Add(testSamples[1]);
                    trainingSet.Add(testSamples[2]);
                    trainingSet.Add(testSamples[3]);

                    #region Initalizations
                    Neuron[] HiddenLayer = new Neuron[hiddenLayerCount];
                    Neuron[] OutputLayer = new Neuron[4];
                    for (int i = 0; i < HiddenLayer.Length; i++)
                    {
                        HiddenLayer[i] = new Neuron(4, 1);
                        HiddenLayer[i].RandomizeValues();

                    }
                    for (int i = 0; i < OutputLayer.Length; i++)
                    {
                        OutputLayer[i] = new Neuron(hiddenLayerCount, 1);
                        OutputLayer[i].RandomizeValues();
                    }
                    #endregion

                    do
                    {
                        MSE = 0;
                        ErasMSE = 0;
                        for (int i = trainingSet.Count; i > 0; i--)
                        {
                            int randomIndex = gen.Next(i);

                            double[] inputs1 = trainingSet[randomIndex];
                            double[] inputs2 = new double[hiddenLayerCount];
                            foreach (Neuron n in HiddenLayer)
                            {
                                n.Inputs = inputs1;
                            }
                            for (int j = 0; j < HiddenLayer.Length; j++)
                            {
                                inputs2[j] = HiddenLayer[j].Output();
                            }
                            foreach (Neuron n in OutputLayer)
                            {
                                n.Inputs = inputs2;
                            }

                            double[] OutputsError = new double[4];

                            for (int j = 0; j < OutputLayer.Length; j++)
                            {
                                OutputsError[j] = (inputs1[j] - OutputLayer[j].Output());
                                MSE += Pow(OutputsError[j], 2);

                            }
                            for (int j = 0; j < OutputLayer.Length; j++)
                            {
                                OutputLayer[j].Error = Sigm.FunctionDerivative(OutputLayer[j].Output()) * (OutputsError[j]);
                                OutputLayer[j].UpdateWeights();
                            }

                            for (int j = 0; j < HiddenLayer.Length; j++)
                            {
                                double value = 0;
                                for (int k = 0; k < HiddenLayer[j].Weights.Length; k++)
                                {
                                    value += Sigm.FunctionDerivative(HiddenLayer[j].Output()) * OutputLayer[k].Error * OutputLayer[k].Weights[j];
                                }
                                HiddenLayer[j].Error = value;
                            }
                            for (int j = 0; j < HiddenLayer.Length; j++)
                            {
                                HiddenLayer[j].UpdateWeights();
                            }
                            trainingSet.RemoveAt(randomIndex);
                            ErasMSE += MSE;
                            if (iteration == erasCount && i == 1)
                            {
                                double[][] outputs = new double[2][];
                                XmlSerializer xs1 = new XmlSerializer(typeof(double[][]));
                                outputs[0] = inputs1;
                                outputs[1] = new double[] { OutputLayer[0].Output(), OutputLayer[1].Output(), OutputLayer[2].Output(), OutputLayer[3].Output() };
                                string fileName1 = fileName;
                                fileName1 = fileName.Replace(".xml", "");
                                using (StreamWriter sw1 = new StreamWriter(fileName1 + "output.xml"))
                                {
                                    xs1.Serialize(sw1, outputs);
                                }
                            }
                        }
                        ErasMSE /= 4;
                        trainingSet.Add(testSamples[0]);
                        trainingSet.Add(testSamples[1]);
                        trainingSet.Add(testSamples[2]);
                        trainingSet.Add(testSamples[3]);
                        if (iteration % 20 == 1)
                            toSerialize.Add(ErasMSE);
                        iteration++;
                    } while (iteration <= erasCount);
                    for (int j = 0; j < OutputLayer.Length; j++)
                    {
                        Write(OutputLayer[j].Output() + " ");
                    }
                    WriteLine();
                    xs.Serialize(sw, toSerialize);
                }
            }

            #endregion

            #region Aproksymacja
            if (variant == 2)
            {
                for (int counter = 1; counter <= repetitions; counter++)
                {
                    string fileName = "APP1Neuron0100@" + counter.ToString() + ".xml";
                    StreamWriter sw = new StreamWriter(fileName);
                    XmlSerializer xs = new XmlSerializer(typeof(List<ApproximationData>));
                    List<ApproximationData> toSerialize = new List<ApproximationData>();
                    #region LoadData
                    StreamReader streamReader = new StreamReader("approximation_train_1.txt");
                    string string1 = streamReader.ReadToEnd();
                    streamReader = new StreamReader("approximation_train_2.txt");
                    string string2 = streamReader.ReadToEnd();
                    string1 = string1.Replace('\n', ' ');
                    string2 = string2.Replace('\n', ' ');
                    string1 = string1.Replace('.', ',');
                    string2 = string2.Replace('.', ',');
                    string[] traningSet1 = string1.Split(' ');
                    string[] trainingSet2 = string2.Split(' ');
                    List<double> input = new List<double>();
                    List<double> output = new List<double>();
                    if (trainingSetNumber == 1)
                    {
                        for (int i = 0; i < traningSet1.Length - 1; i++)
                        {
                            if (i % 2 == 0)
                            {
                                input.Add(double.Parse(traningSet1[i]));
                            }
                            else
                            {
                                output.Add(double.Parse(traningSet1[i]));
                            }
                        }
                    }
                    if (trainingSetNumber == 2)
                    {
                        for (int i = 0; i < trainingSet2.Length - 1; i++)
                        {
                            if (i % 2 == 0)
                            {
                                input.Add(double.Parse(trainingSet2[i]));
                            }
                            else
                            {
                                output.Add(double.Parse(trainingSet2[i]));
                            }
                        }
                    }
                    streamReader = new StreamReader("approximation_test.txt");
                    string string3 = streamReader.ReadToEnd();
                    string3 = string3.Replace('\n', ' ');
                    string3 = string3.Replace(Environment.NewLine, " ");
                    string3 = string3.Replace('.', ',');
                    string[] testingSet = string3.Split(' ');
                    List<double> testingSetInput = new List<double>();
                    List<double> testingSetOutput = new List<double>();

                    for (int i = 0; i < testingSet.Length - 1; i++)
                    {
                        if (i % 2 == 0)
                        {
                            testingSetInput.Add(double.Parse(testingSet[i]));
                        }
                        else
                        {
                            testingSetOutput.Add(double.Parse(testingSet[i]));
                        }
                    }
                    #endregion

                    #region Initalization
                    Neuron[] HiddenLayer = new Neuron[hiddenLayerCount];
                    Neuron[] OutputLayer = new Neuron[1];
                    for (int i = 0; i < HiddenLayer.Length; i++)
                    {
                        HiddenLayer[i] = new Neuron(1, 1);
                        HiddenLayer[i].RandomizeValues();

                    }
                    OutputLayer[0] = new Neuron(hiddenLayerCount, 2);
                    OutputLayer[0].RandomizeValues();
                    int iteration = 1;
                    double MSE = 0;
                    #endregion
                    do
                    {
                        List<int> numbers = new List<int>();
                        List<double> finalOutputs = new List<double>();
                        for (int i = 0; i < input.Count; i++)
                        {
                            numbers.Add(i);
                        }
                        MSE = 0;
                        for (int i = input.Count; i > 0; i--)
                        {
                            int randomIndex = gen.Next(i);
                            double[] inputs1 = new double[] { input[randomIndex] };
                            double[] inputs2 = new double[hiddenLayerCount];
                            foreach (Neuron n in HiddenLayer)
                            {
                                n.Inputs = inputs1;
                            }
                            for (int j = 0; j < HiddenLayer.Length; j++)
                            {
                                inputs2[j] = HiddenLayer[j].Output();
                            }

                            OutputLayer[0].Inputs = inputs2;

                            double diffrence = 0;
                            diffrence = output[randomIndex] - OutputLayer[0].Output();
                            MSE += Pow(diffrence, 2);

                            OutputLayer[0].Error = Linear.FunctionDerivative(OutputLayer[0].Output()) * diffrence;
                            for (int j = 0; j < HiddenLayer.Length; j++)
                            {
                                HiddenLayer[j].Error = Sigm.FunctionDerivative(HiddenLayer[j].Output()) * OutputLayer[0].Error * OutputLayer[0].Weights[j];
                                HiddenLayer[j].UpdateWeights();
                            }

                            OutputLayer[0].UpdateWeights();

                            numbers.RemoveAt(randomIndex);
                        }
                        MSE /= input.Count;
                        #region CheckNetworkUsingTestingSet
                        double TestMSE = 0;
                        for (int j = 0; j < testingSetInput.Count; j++)
                        {
                            double[] inputs1 = new double[] { testingSetInput[j] };
                            double[] inputs2 = new double[hiddenLayerCount];
                            foreach (Neuron n in HiddenLayer)
                            {
                                n.Inputs = inputs1;
                            }
                            for (int k = 0; k < HiddenLayer.Length; k++)
                            {
                                inputs2[k] = HiddenLayer[k].Output();
                            }

                            OutputLayer[0].Inputs = inputs2;

                            TestMSE += Pow(testingSetOutput[j] - OutputLayer[0].Output(), 2);
                            if (iteration == erasCount)
                            {
                                finalOutputs.Add(OutputLayer[0].Output());
                            }
                        }
                        if (iteration == erasCount)
                        {
                            string fileName1 = fileName;
                            fileName1 = fileName.Replace(".xml", "");
                            XmlSerializer xs1 = new XmlSerializer(typeof(List<double>));
                            using (StreamWriter sw1 = new StreamWriter(fileName1 + "wyjscie.xml"))
                            {
                                xs1.Serialize(sw1, finalOutputs);
                            }
                        }
                        TestMSE /= testingSetInput.Count;
                        ApproximationData approximationData;
                        approximationData.MSETrening = MSE;
                        approximationData.MSETest = TestMSE;
                        if (iteration % 20 == 1)
                            toSerialize.Add(approximationData);
                        #endregion
                        iteration++;
                    } while (iteration <= erasCount);

                    xs.Serialize(sw, toSerialize);
                }
            }
            #endregion
        }
    }
}