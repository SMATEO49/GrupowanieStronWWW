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


namespace GrupowanieStronWWW
{
    public partial class MainWindow : Window
    {
        private List<CityData> _cityData;
        //private List<AnalizedCity> _analizedData;
        private List<ClusteredCity> _clusteredData;
        private static readonly string[] collection = new[] { "the", "and", "is", "of", "in", "to", "a", "on", "for", "as", "with", "by", "it", "at" };

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
                    CityListView.ItemsSource = _cityData;

                    MessageBox.Show("Data loaded successfully!");
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error loading data: {ex.Message}");
            }
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
        private void AnalyzeButton_Click(object sender, RoutedEventArgs e)
        {
            /*
            if (_cityData == null || _cityData.Count == 0)
            {
                MessageBox.Show($"Error analizing data: no data loaded");
                return;
            }
            try
            {

                _analizedData = new List<AnalizedCity>();

                foreach (var cityData in _cityData)
                {
                    var tokens = cityData.Description != null ? Tokenize(cityData.Description) : new List<string>();
                    var analyzedCity = new AnalizedCity(cityData.Country, cityData.City, tokens);
                    _analizedData.Add(analyzedCity);

                }
                CityListAnalyzed.ItemsSource = _analizedData;
                MessageBox.Show("Data analized successfully!");
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error analizing data: {ex.Message}");
            }*/
        }


        private void RunClusteringButton_Click(object sender, RoutedEventArgs e)
        {
            if (_cityData == null || !_cityData.Any())
            {
                MessageBox.Show("No data loaded!");
                return;
            }

            var selectedMethod = (EncodingMethodComboBox.SelectedItem as ComboBoxItem)?.Content.ToString();
            var descriptions = _cityData.Select(c => c.Description).ToList();
            double[][] featureVectors;
            switch (selectedMethod)
            {
                case "TF":
                    featureVectors = ComputeTF(descriptions);
                    break;
                case "TF-IDF":
                    featureVectors = ComputeTFIDF(descriptions);
                    break;
                case "Binary":
                    featureVectors = ComputeBinary(descriptions);
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

                switch (selectedAlgorithm)
                {
                    case "K-Means":
                        RunKMeans(featureVectors, clusters: clusters, distanceMetric: distanceMetrics);
                        break;
                    //case "EM":
                        //RunEMClustering(featureVectors, metric, clusters);
                        //break;
                    case "Hierarchical":
                        RunHierarchical(featureVectors, clusters, metric);
                        break;
                    //case "DBSCAN":
                        //RunDBSCAN(featureVectors, distanceMetrics,epsilon, clusters);
                        //break;
                    //case "Mean-Shift":
                        //RunMeanShift(featureVectors);
                        //break;
                    default:
                        MessageBox.Show("No valid algorithm selected.");
                        break;
                }
            }
        }

        private void RunHierarchical(double[][] featureVectors, int desiredClusters, string metric)
        {
            //Debug.WriteLine("Input Feature Vectors:");
            //foreach (var vector in featureVectors)
            //{
                //Debug.WriteLine($"({string.Join(", ", vector)})");
            //}
            if (metric != "Cosine" & metric != "Euclidean") throw new ArgumentException("Unsupported distance metric. Use 'Euclidean' or 'Cosine'.");

            List<List<double[]>> clusters = featureVectors.Select(fv => new List<double[]> {fv}).ToList();
            List<int> clustersBook = Enumerable.Range(0, featureVectors.Length).ToList();
            var lenght = featureVectors.Length;
            var clustersCount = featureVectors.Length;

            if (metric == "Cosine")
            {
                while (clustersCount > desiredClusters)
                {
                    double minDistance = double.MaxValue;
                    int cluster1Index = 0;
                    int cluster2Index = 0;

                    for (int i = 0; i + 1 < lenght; i++)
                    {
                        for (int j = i+1; j < lenght; j++)
                        {
                            if (clustersBook[i] != clustersBook[j])
                            {
                                double distance = CosineDistance(featureVectors[i], featureVectors[j]);//CalculateCosineClusterDistance(clusters[i], clusters[j]);

                                if (distance < minDistance)
                                {
                                    minDistance = distance;
                                    cluster1Index = i;
                                    cluster2Index = j;
                                }
                            }
                        }
                    }
                    Debug.WriteLine(minDistance.ToString());
                    // laczenie klastrow tak aby zachowac najmniejsze mozliwie numery grup
                    if (clustersBook[cluster1Index] < clustersBook[cluster2Index])
                    {
                        for(int i = 0;i < lenght;i++)
                        {
                            if (clustersBook[i] == clustersBook[cluster2Index]) clustersBook[i] = clustersBook[cluster1Index];
                        }
                    }
                    else
                    {
                        for (int i = 0; i < lenght; i++)
                        {
                            if (clustersBook[i] == clustersBook[cluster1Index]) clustersBook[i] = clustersBook[cluster2Index];
                        }
                    }

                    clustersCount = clustersBook.Distinct().Count();
                    Debug.WriteLine(clustersCount.ToString());
                    Debug.WriteLine($"{clustersBook[cluster1Index]} {clustersBook[cluster2Index]} {cluster1Index} {cluster2Index}");
                }
            }
            else
            {
                while (clusters.Count > desiredClusters)
                {
                    double minDistance = double.MaxValue;
                    int cluster1Index = 0;
                    int cluster2Index = 0;

                    for (int i = 0; i + 1 < clusters.Count; i++)
                    {
                        for (int j = +1; j < clusters.Count; j++)
                        {
                            double distance = CalculateEuclideanClusterDistance(clusters[i], clusters[j]);
                            if (distance < minDistance)
                            {
                                minDistance = distance;
                                cluster1Index = i;
                                cluster2Index = j;
                            }

                        }
                    }
                    clusters[cluster1Index].AddRange(clusters[cluster2Index]);
                    clusters.RemoveAt(cluster2Index);
                }
            }

            Debug.WriteLine("Moje grupy");
            for (int i = 0; i < lenght; i++)
            {
                Debug.WriteLine(clustersBook[i].ToString());
            }


            MessageBox.Show("Hierarchical clustering completed!");
        }

        class ArrayComparer : IEqualityComparer<double[]>
        {
            public bool Equals(double[] x, double[] y)
            {
                return x.SequenceEqual(y);
            }

            public int GetHashCode(double[] obj)
            {
                unchecked
                {
                    return obj.Aggregate(17, (hash, value) => hash * 31 + value.GetHashCode());
                }
            }
        }

        private double CalculateCosineClusterDistance(List<double[]> cluster1,  List<double[]> cluster2)
        {
            double minDistance = double.MaxValue;

            foreach (var point1 in cluster1)
            {
                foreach (var point2 in cluster2)
                {
                    double distance = CosineDistance(point1, point2);
                    if (distance < minDistance)
                    {
                        minDistance = distance;
                    }
                }
            }

            return minDistance;
        }
        private double CalculateEuclideanClusterDistance(List<double[]> cluster1, List<double[]> cluster2)
        {
            double minDistance = double.MaxValue;

            foreach (var point1 in cluster1)
            {
                foreach (var point2 in cluster2)
                {
                    double distance = EuclideanDistance(point1, point2);
                    if (distance < minDistance)
                    {
                        minDistance = distance;
                    }
                }
            }

            return minDistance;
        }


        private void RunEMClustering(double[][] featureVectors, string metric, int clusters)
        {
            try
            {
                if (metric == "Cosine")
                {
                    featureVectors = NormalizeData(featureVectors);
                    MessageBox.Show("EM cosine clustering completed!");
                }
                else
                {
                    MessageBox.Show("EM euclidean clustering completed!");
                }

                var gmm = new GaussianMixtureModel(clusters);

                var labels = gmm.Learn(featureVectors).Decide(featureVectors);

                _clusteredData = _cityData.Select((city, index) => new ClusteredCity
                {
                    Country = city.Country,
                    City = city.City,
                    Description = city.Description,
                    GroupLabel = labels[index]
                }).ToList();

                ResultsListView.ItemsSource = _clusteredData;
                MessageBox.Show("EM clustering completed!");
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error running EM Clustering: {ex.Message}");
            }
        }

        private double[][] NormalizeData(double[][] data)
        {
            return data.Select(row =>
            {
                double norm = Math.Sqrt(row.Sum(value => value * value));
                return row.Select(value => value / norm).ToArray();
            }).ToArray();
        }


        /*
        private void RunHierarchicalClustering(double[][] featureVectors, int clusters)
        {
                //var hierarchical = new AgglomerativeClustering()
                {
                    Distance = new Euclidean(),      // Use Euclidean distance
                    Linkage = Linkage.Complete      // Use Complete Linkage
                };

                // Step 2: Perform clustering
                int[] labels = hierarchical.Learn(featureVectors).Decide(featureVectors, clusters);

                // Step 3: Map results back to the city data
                _clusteredData = _cityData.Select((city, index) => new ClusteredCity
                {
                    Country = city.Country,
                    City = city.City,
                    Description = city.Description,
                    GroupLabel = labels[index]
                }).ToList();

                // Step 4: Display results in the ListView
                ResultsListView.ItemsSource = _clusteredData;

                MessageBox.Show("Hierarchical clustering completed!");
        }*/

        private void RunKMeans(double[][] featureVectors, int clusters, IDistance<double[]> distanceMetric)
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

            ResultsListView.ItemsSource = _clusteredData;
            MessageBox.Show("K-Means clustering completed!");
        }

        /*
        private void RunDBSCAN(double[][] featureVectors, IDistance<double[]> distanceMetric, double epsilon, int clusters)
        {
            try
            {
                var dbscan = new DBSCAN<IDistance<double[]>>(epsilon: epsilon, minPoints: clusters, distance: distanceMetric);


                var labels = dbscan.Compute(featureVectors);

                _clusteredData = _cityData.Select((c, i) => new ClusteredCity
                {
                    Country = c.Country,
                    City = c.City,
                    Description = c.Description,
                    GroupLabel = labels[i] == -1 ? "Outlier" : labels[i].ToString()
                }).ToList();

                ResultsListView.ItemsSource = _clusteredData;
                MessageBox.Show("DBSCAN clustering completed!");
            }
            catch
            (Exception ex)
            {
                MessageBox.Show($"Error running DBSCAN: {ex.Message}");
            }                     
        }*/

        private double[][] ComputeBinary(List<string> descriptions)
        {
            var terms = descriptions.Select(d => Tokenize(d)).ToList();
            var allTerms = terms.SelectMany(t => t).Distinct().ToList();

            double[][] binary = new double[descriptions.Count][];
            for (int i = 0; i < descriptions.Count; i++)
            {
                binary[i] = new double[allTerms.Count];
                for (int j = 0; j < allTerms.Count; j++)
                {
                    var term = allTerms[j];
                    binary[i][j] = terms[i].Contains(term) ? 1.0 : 0.0;
                }
            }
            return binary;
        }
        private double[][] ComputeTF(List<string> descriptions)
        {
            var terms = descriptions.Select(d => Tokenize(d)).ToList();
            var allTerms = terms.SelectMany(t => t).Distinct().ToList();

            double[][] tf = new double[descriptions.Count][];
            for (int i = 0; i < descriptions.Count; i++)
            {
                tf[i] = new double[allTerms.Count];
                for (int j = 0; j < allTerms.Count; j++)
                {
                    var term = allTerms[j];
                    tf[i][j] = terms[i].Count(t => t == term) / (double)terms[i].Count;
                }
            }
            return tf;
        }
        private double[][] ComputeTFIDF(List<string> descriptions)
        {
            var terms = descriptions.Select(d => Tokenize(d)).ToList();
            var allTerms = terms.SelectMany(t => t).Distinct().ToList();

            var tfidf = new double[descriptions.Count][];
            for (int i = 0; i < descriptions.Count; i++)
            {
                tfidf[i] = new double[allTerms.Count];
                for (int j = 0; j < allTerms.Count; j++)
                {
                    var term = allTerms[j];
                    tfidf[i][j] = ComputeTFIDFForTerm(terms, term, i);
                }
            }
            return tfidf;
        }
        private double ComputeTFIDFForTerm(List<List<string>> terms, string term, int documentIndex)
        {
            double tf = terms[documentIndex].Count(t => t == term) / (double)terms[documentIndex].Count;
            double idf = Math.Log(terms.Count / (1 + terms.Count(d => d.Contains(term))));
            return tf * idf;
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
    }
}