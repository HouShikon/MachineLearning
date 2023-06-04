using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.Intrinsics.X86;

namespace IrisClassification
{
    class IrisSample
    {
        public List<double> Features { get; set; }
        public string Class { get; set; }

        public IrisSample(List<double> features, string className)
        {
            Features = features;
            Class = className;
            
        }
    }
    class Program
    {
        static void Main(string[] args)
        {
            int n = 20; // Number of experiments

            double knnAccuracySum = 0;
            List<double> knnAccuracyList = new List<double>(20);
            double fisherAccuracyVarianceSum = 0;

            for (int exp = 0; exp < n; exp++)
            {
                List<IrisSample> irisData = LoadIrisData("iris.data");
                List<IrisSample> trainingSet = new List<IrisSample>();
                List<IrisSample> testSet = new List<IrisSample>();
                Random random = new Random();
                foreach (var iris in irisData)
                {
                    if (random.NextDouble() < 0.5)
                        trainingSet.Add(iris);
                    else
                        testSet.Add(iris);
                }

                // Perform classification using k-nearest neighbor
                int k = 5; // Number of nearest neighbors to consider

                int correctClassificationCount = 0;
                foreach (var testSample in testSet)
                {
                    string predictedClass = KNearestNeighbor(trainingSet, testSample, k);
                    if (predictedClass == testSample.Class)
                        correctClassificationCount++;
                }
                double knnAccuracy = (double)correctClassificationCount / testSet.Count;
                knnAccuracySum += knnAccuracy;
                knnAccuracyList.Add(knnAccuracy);
                             
                double fisherAccuracyVariance = fisherAccuracyVarianceSum / (n * testSet.Count);

                /*Console.WriteLine("Experiment"+exp+" k-Nearest Neighbor:");
                Console.WriteLine("Experiment" + exp + " Accuracy: " + knnAccuracy);

                Console.WriteLine("^^^^^^^^^^^^^^^^^^^^^^^^");*/
            }
            double knnMeanAccuracy = knnAccuracySum / n;
            var knnVariance = knnAccuracyList.Sum(x => Math.Pow(x - knnMeanAccuracy, 2)) / knnAccuracyList.Count();
            Console.WriteLine("KNN Mean Accuracy:"+knnMeanAccuracy);
            Console.WriteLine("KNN Variance:" + knnVariance);
            Console.WriteLine("^^^^^^^^^^^^^^^^^^^^^^^^"); 

        }

       


        static string KNearestNeighbor(List<IrisSample> trainingSet, IrisSample testSample, int k)
        {

            var distanceInsample = new List<Tuple<string, double>>();
            Dictionary<string, int> classVotes = new Dictionary<string, int>();
            foreach (var trainingSample in trainingSet)
            {
                double distance = EuclideanDistance(trainingSample.Features, testSample.Features);
                distanceInsample.Add(Tuple.Create ( trainingSample.Class, distance));
                
            }
            distanceInsample = distanceInsample.OrderBy(x => x.Item2).ToList();
            int i = 0;
            foreach(var item in distanceInsample)
            {
                if(i==k)
                {
                    break;
                }
                if(!classVotes.ContainsKey(item.Item1))
                {
                    classVotes.Add(item.Item1, 1);
                }
                else
                {
                    classVotes[item.Item1]++;
                }
                i++;

            }

            int maxVotes = -1;
            string predictedClass = "";
            foreach (var vote in classVotes)
            {
                if (vote.Value > maxVotes)
                {
                    maxVotes = vote.Value;
                    predictedClass = vote.Key;
                }
            }

            return predictedClass;
        }


        static double EuclideanDistance(List<double> features1, List<double> features2)
        {
            double distance = 0;
            for (int i = 0; i < features1.Count; i++)
            {
                distance += Math.Pow(features1[i] - features2[i], 2);
            }
            return Math.Sqrt(distance);
        }

        static List<IrisSample> LoadIrisData(string filePath)
        {
            List<IrisSample> irisData = new List<IrisSample>();

            using (StreamReader reader = new StreamReader(filePath))
            {
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    string[] parts = line.Split(',');
                    if (parts.Length == 5)
                    {
                        List<double> features = new List<double>();
                        for (int i = 0; i < 4; i++)
                        {
                            features.Add(double.Parse(parts[i]));
                        }

                        string className = parts[4];

                        IrisSample sample = new IrisSample(features, className);
                        irisData.Add(sample);
                    }
                }
            }

            return irisData;
        }
    }

    
}
