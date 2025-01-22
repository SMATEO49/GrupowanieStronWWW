using Accord.MachineLearning;
using Accord.Math.Distances;
using CsvHelper;
using CsvHelper.Configuration;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Windows;
using System.Windows.Controls;
using Accord.Statistics.Distributions.Multivariate;
using System.Diagnostics;
using Numerics.NET.Data.Stata;
using Numerics.NET;
using Numerics.NET.Statistics;
using Numerics.NET.Statistics.Multivariate;
using static GrupowanieStronWWW.MainWindow;
using System.Reflection.Emit;


namespace GrupowanieStronWWW
{
    public partial class MainWindow : Window
    {
        private List<CityData> _cityData;
        private List<ClusteredCity> _clusteredData;
        private static readonly string[] collection = new[] { "the", "and", "is", "of", "in", "to", "a", "on", "for", "as", "with", "by", "it", "at" };
        private List<string> allStrings;
        private int[] trueLabels;

        public class CityData
        {
            public required string Country { get; set; }
            public required string City { get; set; }
            public required string Description { get; set; }
        }
        public class ClusteredCity : CityData
        {
            public int GroupLabel { get; set; }
        }
        public class AnalizedCity
        {
            public string Country { get; set; }
            public string City { get; set; }
            public List<string> ListOfTokens { get; set; }
            public string JoinedTokens => string.Join(", ", ListOfTokens);
            public AnalizedCity(string country, string city, List<string> listOfTokens)
            {
                this.Country = country;
                this.City = city;
                this.ListOfTokens = listOfTokens;
            }
        }

        public MainWindow()
        {
            InitializeComponent();
            const string license = "xxxx";
            Numerics.NET.License.Verify(license);

        }
        private void LoadButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                var openFileDialog = new Microsoft.Win32.OpenFileDialog
                {
                    Filter = "CSV Files (*.csv)|*.csv"
                };

                if (openFileDialog.ShowDialog() == true)
                {
                    using var reader = new StreamReader(openFileDialog.FileName);
                    var config = new CsvConfiguration(CultureInfo.InvariantCulture)
                    {
                        BadDataFound = null // HeaderValidated = null, // Disable header validation //MissingFieldFound = null // Disable missing field validation
                    };
                    using var csv = new CsvReader(reader, config);

                    _cityData = csv.GetRecords<CityData>().ToList();

                    trueLabels = GenerateLabelsByClass(_cityData, city => city.Country);

                    CityListView.ItemsSource = _cityData;

                    MessageBox.Show("Data loaded successfully!");
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error loading data: {ex.Message}");
            }
        }

        private int[] GenerateLabelsByClass<T>(List<T> data, Func<T, string> classSelector)
        {
            var classToClusterMap = data
                .Select(classSelector)
                .Distinct()
                .Select((className, index) => new { className, index })
                .ToDictionary(x => x.className, x => x.index);

            return data
                .Select(item => classToClusterMap[classSelector(item)])
                .ToArray();
        }
        private static List<string> Tokenize(string text)
        {
            string workText = new(text.ToLower());
            workText = Regex.Replace(workText, @"[^\w\s]", "");
            workText = Regex.Replace(workText, @"\d+", "");
            var stopwords = new HashSet<string>(collection);
            List<string> result = workText.Split(' ').Where(word => !stopwords.Contains(word) && !string.IsNullOrWhiteSpace(word)).ToList();
            return result;
        }
        private void RunClusteringButton_Click(object sender, RoutedEventArgs e)
        {
            if (_cityData == null || !_cityData.Any())
            {
                MessageBox.Show("No data loaded!");
                return;
            }

            var selectedMethod = EncodingMethodComboBox.Text;
            var descriptions = _cityData.Select(c => c.Description).ToList();
            
            var longSentence = string.Join(" ", descriptions);

            var terms = Tokenize(longSentence);
            allStrings = terms.Distinct().ToList();
            
            double[][] featureVectors;
            switch (selectedMethod)
            {
                case "TF":
                    featureVectors = ComputeTF(descriptions, allStrings);
                    break;
                case "TF-IDF":
                    featureVectors = ComputeTFIDF(descriptions, allStrings);
                    break;
                case "Binary":
                    featureVectors = ComputeBinary(descriptions, allStrings);
                    break;
                default:
                    MessageBox.Show("Please select an encoding method.");
                    return;
            }

            var dialog = new AlgorithmSelectionWindow();
            if (dialog.ShowDialog() == true)
            {
                var selectedAlgorithm = dialog.SelectedAlgorithm;
                var clusters = dialog.Clusters;
                var epsilon = dialog.Epsilon;
                var distanceMetrics = dialog.DistanceMetrics;
                var metric = dialog.metric;
                var linkage = dialog.linkage;

                switch (selectedAlgorithm)
                {
                    case "K-Means":
                        RunKMeans(featureVectors, clusters: clusters, distanceMetric: distanceMetrics, metric: metric);
                        break;
                    case "Hierarchical":
                        RunHierarchical(featureVectors, clusters, metric, linkage);
                        break;
                    default:
                        MessageBox.Show("No valid algorithm selected.");
                        break;
                }
            }
        }

        private void RunHierarchical(double[][] featureVectors, int desiredClusters, string metric, string linkage)
        {
            if (metric != "Cosine" & metric != "Euclidean") throw new ArgumentException("Unsupported distance metric. Use 'Euclidean' or 'Cosine'.");            
            int length = featureVectors.Length;
            List<int> clustersBook = Enumerable.Range(0, length).ToList();
            var clustersCount = length;
            ValidateFeatureVectors(featureVectors);
            var dataMatrix = Matrix.Create(featureVectors);
            var hc = new HierarchicalClusterAnalysis(dataMatrix);
            
            if (metric == "Cosine")
            {
                hc.DistanceMeasure = DistanceMeasures.CosineDistance;
            }
            else
            {
                hc.DistanceMeasure = DistanceMeasures.EuclideanDistance;
            }

            switch (linkage)
            {
                case "McQuitty":
                    hc.LinkageMethod = LinkageMethod.McQuitty;
                    break;
                case "Median":
                    hc.LinkageMethod = LinkageMethod.Median;
                    break;
                case "Average":
                    hc.LinkageMethod = LinkageMethod.Average;
                    break;
                case "Centroid":
                    hc.LinkageMethod = LinkageMethod.Centroid;
                    break;
                case "Ward":
                    hc.LinkageMethod = LinkageMethod.Ward;
                    break;
                case "Single":
                    hc.LinkageMethod = LinkageMethod.Single;
                    break;
                case "Complete":
                    hc.LinkageMethod = LinkageMethod.Complete;
                    break;
                default:
                    break;
            }

            hc.Standardize = true;
            hc.Fit();
            var partition = hc.GetClusterPartition(desiredClusters);
            foreach (var cluster in partition)
            {
                Debug.WriteLine($"Cluster {cluster.Index} has {cluster.Size} members.");
            }

            var memberships = partition.GetMemberships();
            int[] labels = new int[length];
            _clusteredData = _cityData.Select((c, i) => new ClusteredCity
            {
                Country = c.Country,
                City = c.City,
                Description = c.Description,
                GroupLabel = memberships.GetLevelIndex(i),
                
            }).ToList();
            for (int i = 0; i < length; i++)
            {
                labels[i] = memberships.GetLevelIndex(i);
            }

            ResultsListView.ItemsSource = _clusteredData;
            double RandId = RandIndex(trueLabels, labels);
            double silhouetteScore = SilhouetteScore(featureVectors, labels, metric == "Cosine"? CosineDistance : EuclideanDistance);
            double wcss = WCSS(featureVectors, labels, metric == "Cosine" ? CosineDistance : EuclideanDistance);
            ResultsListView.ItemsSource = _clusteredData;
            MessageBox.Show($"K-Means clustering completed!\nRandIndex: \t{RandId:F4}\nSilhouetteScore: \t{silhouetteScore:F4}\nWCSS: \t\t{wcss:F4}");
        }
        
        private void RunKMeans(double[][] featureVectors, int clusters, IDistance<double[]> distanceMetric, string metric)
        {


            var kMeans = new Accord.MachineLearning.KMeans(k: clusters, distance: distanceMetric);
            var labels = kMeans.Learn(featureVectors).Decide(featureVectors);

            _clusteredData = _cityData.Select((c, i) => new ClusteredCity
            {
                Country = c.Country,
                City = c.City,
                Description = c.Description,
                GroupLabel = labels[i]
            }).ToList();

            double RandId = RandIndex(trueLabels,labels);
            double silhouetteScore = SilhouetteScore(featureVectors, labels, metric == "Cosine" ? CosineDistance : EuclideanDistance);
            double wcss = WCSS(featureVectors,labels, metric == "Cosine" ? CosineDistance : EuclideanDistance);
            ResultsListView.ItemsSource = _clusteredData;
            MessageBox.Show($"K-Means clustering completed!\nRandIndex: \t{RandId:F4}\nSilhouetteScore: \t{silhouetteScore:F4}\nWCSS: \t\t{wcss:F4}");
        }

        private void ValidateFeatureVectors(double[][] featureVectors)
        {
            if (featureVectors == null || featureVectors.Length == 0)
                throw new ArgumentException("Feature vectors cannot be null or empty.");

            int firstRowLength = featureVectors[0].Length;
            foreach (var row in featureVectors)
            {
                if (row.Length != firstRowLength)
                    throw new ArgumentException("All rows in feature vectors must have the same length.");
            }
        }       
        private double[][] ComputeBinary(List<string> descriptions, List<string> allStrings)
        {
            var tokenizedDescriptions = descriptions.Select(desc => Tokenize(desc)).ToList();

            double[][] binaryMatrix = new double[descriptions.Count][];
            for (int i = 0; i < descriptions.Count; i++)
            {
                binaryMatrix[i] = new double[allStrings.Count];
                foreach (var term in allStrings)
                {
                    binaryMatrix[i][allStrings.IndexOf(term)] = tokenizedDescriptions[i].Contains(term) ? 1.0 : 0.0;
                }
            }
            return binaryMatrix;
        }
        private double[][] ComputeTF(List<string> descriptions, List<string> allStrings)
        {
            // Tokenize descriptions
            var tokenizedDescriptions = descriptions.Select(desc => Tokenize(desc)).ToList();
            double[][] tfMatrix = new double[descriptions.Count][];

            for (int i = 0; i < descriptions.Count; i++)
            {
                // Handle empty tokenized descriptions
                if (tokenizedDescriptions[i] == null || tokenizedDescriptions[i].Count == 0)
                {
                    tfMatrix[i] = new double[allStrings.Count]; // Fill with zeros
                    continue;
                }

                tfMatrix[i] = new double[allStrings.Count];
                foreach (string s in allStrings)
                {
                    // Ensure the term exists in tokenized descriptions
                    double termFrequency = tokenizedDescriptions[i].Count(t => t == s) / (double)tokenizedDescriptions[i].Count;
                    int termIndex = allStrings.IndexOf(s);
                    if (termIndex >= 0)
                    {
                        tfMatrix[i][termIndex] = termFrequency;
                    }
                }
            }
            return tfMatrix;
        }
        private double[][] ComputeTFIDF(List<string> descriptions, List<string> allStrings)
        {
            var tokenizedDescriptions = descriptions.Select(desc => Tokenize(desc)).ToList();
            double[][] tfidfMatrix = new double[descriptions.Count][];
            // Compute IDF for each term
            var idfValues = new double[allStrings.Count];
            for (int j = 0; j < allStrings.Count; j++)
            {
                string term = allStrings[j];
                int documentCountWithTerm = tokenizedDescriptions.Count(doc => doc.Contains(term));
                idfValues[j] = Math.Log((1 + descriptions.Count) / (1 + documentCountWithTerm));
            }
            // Compute TF-IDF for each document
            for (int i = 0; i < descriptions.Count; i++)
            {
                tfidfMatrix[i] = new double[allStrings.Count];
                foreach (var term in allStrings)
                {
                    double tf = tokenizedDescriptions[i].Count(t => t == term) / (double)tokenizedDescriptions[i].Count;
                    double idf = idfValues[allStrings.IndexOf(term)];
                    tfidfMatrix[i][allStrings.IndexOf(term)] = tf * idf;
                }
            }


            return tfidfMatrix;
        }
        private void SaveResultsButton_Click(object sender, RoutedEventArgs e)
        {
            if (_clusteredData == null || !_clusteredData.Any())
            {
                MessageBox.Show("No results to save!");
                return;
            }

            var saveFileDialog = new Microsoft.Win32.SaveFileDialog
            {
                Filter = "CSV Files (*.csv)|*.csv",
                FileName = "ClusteringResults.csv"
            };

            if (saveFileDialog.ShowDialog() == true)
            {
                using var writer = new StreamWriter(saveFileDialog.FileName);
                using var csv = new CsvWriter(writer, CultureInfo.InvariantCulture);
                csv.WriteRecords(_clusteredData);
                MessageBox.Show("Results saved successfully!");
            }
        }
        private static double EuclideanDistance(double[] par1, double[] par2)
        {
            double res = 0.0;
            int len = par1.Length;

            for (int i = 0; i < len; i++)
            {
                res = res + ((par1[i] - par2[i]) * (par1[i] - par2[i]));
            }
            res = Math.Sqrt(res);

            return res;
        }
        private static double CosineDistance(double[] par1, double[] par2)
        {
            double res = 0.0;
            double a = 0.0;
            double b = 0.0;
            int len = par1.Length;

            for(int i = 0; i < len; i++)
            {
                res = res + (par1[i] * par2[i]);

                a = a + (par1[i] * par1[i]);
                b = b + (par2[i] * par2[i]);
            }
            a = Math.Sqrt(a);
            b = Math.Sqrt(b);

            res = res / (a*b);

            return 1.0 - res;
        }


        public static double RandIndex(int[] trueLabels, int[] predictedLabels)
        {
            int n = trueLabels.Length;
            int tp = 0, tn = 0, fp = 0, fn = 0;

            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    bool sameClass = trueLabels[i] == trueLabels[j];
                    bool sameCluster = predictedLabels[i] == predictedLabels[j];

                    if (sameClass && sameCluster) tp++;
                    else if (!sameClass && !sameCluster) tn++;
                    else if (!sameClass && sameCluster) fp++;
                    else if (sameClass && !sameCluster) fn++;
                }
            }

            return (double)(tp + tn) / (tp + tn + fp + fn);
        }
        public static double SilhouetteScore(double[][] data, int[] labels, Func<double[], double[], double> distanceFunction)
        {
            int n = data.Length;
            double totalSilhouette = 0.0;

            for (int i = 0; i < n; i++)
            {
                double a = 0.0; // Średnia odległość do punktów w tym samym klastrze
                double b = double.MaxValue; // Minimalna średnia odległość do punktów w innych klastrach

                int sameClusterCount = 0;

                for (int j = 0; j < n; j++)
                {
                    if (i == j) continue;

                    double distance = distanceFunction(data[i], data[j]);

                    if (labels[i] == labels[j])
                    {
                        a += distance;
                        sameClusterCount++;
                    }
                    else
                    {
                        // Oblicz odległość do innych klastrów
                        double otherClusterDistance = distance;
                        b = Math.Min(b, otherClusterDistance);
                    }
                }

                a = sameClusterCount > 0 ? a / sameClusterCount : 0.0;

                // Oblicz wskaźnik Silhouette dla punktu i
                double silhouette = (b - a) / Math.Max(a, b);
                totalSilhouette += silhouette;
            }

            // Zwróć średni wskaźnik Silhouette
            return totalSilhouette / n;
        }
        public static double WCSS(double[][] data, int[] labels, Func<double[], double[], double> distanceFunction)
        {
            var centroids = ComputeCentroids(data, labels);

            double wcss = 0.0;

            for (int i = 0; i < data.Length; i++)
            {
                int cluster = labels[i];
                double[] centroid = centroids[cluster];
                wcss += Math.Pow(distanceFunction(data[i], centroid),2);
            }

            return wcss;
        }
        public static Dictionary<int, double[]> ComputeCentroids(double[][] data, int[] labels)
        {
            var clusters = new Dictionary<int, List<double[]>>();

            // Grupuj punkty według etykiet klastrów
            for (int i = 0; i < labels.Length; i++)
            {
                int cluster = labels[i];
                if (!clusters.ContainsKey(cluster))
                    clusters[cluster] = new List<double[]>();

                clusters[cluster].Add(data[i]);
            }

            // Oblicz centroidy jako średnią punktów w każdym klastrze
            var centroids = new Dictionary<int, double[]>();
            foreach (var cluster in clusters)
            {
                int clusterId = cluster.Key;
                var points = cluster.Value;
                int dimensions = points[0].Length;

                double[] centroid = new double[dimensions];
                foreach (var point in points)
                {
                    for (int d = 0; d < dimensions; d++)
                    {
                        centroid[d] += point[d];
                    }
                }

                for (int d = 0; d < dimensions; d++)
                {
                    centroid[d] /= points.Count;
                }

                centroids[clusterId] = centroid;
            }

            return centroids;
        }
    }
}