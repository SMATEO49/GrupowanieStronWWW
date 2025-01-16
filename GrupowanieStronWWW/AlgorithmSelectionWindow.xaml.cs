using System.Windows.Controls;
using System.Windows;
using Accord.Math.Distances;


namespace GrupowanieStronWWW
{
    public partial class AlgorithmSelectionWindow : Window
    {
        public string SelectedAlgorithm { get; private set; }
        public int Clusters { get; private set; }
        public double Epsilon { get; private set; }
        public IDistance<double[]> DistanceMetrics {  get; private set; }

        public string metric {  get; private set; }

        public AlgorithmSelectionWindow()
        {
            InitializeComponent();
        }

        private IDistance<double[]> GetSelectedDistanceMetric()
        {
            var selectedMetric = (DistanceMetricComboBox.SelectedItem as ComboBoxItem)?.Content.ToString();

            if (string.IsNullOrEmpty(selectedMetric))
            {
                throw new InvalidOperationException("Invalid distance metric selected");
            }

            metric = selectedMetric;
            return selectedMetric switch
            {
                "Euclidean" => new Euclidean(),
                "Cosine" => new Cosine(),
                _ => throw new InvalidOperationException("Invalid distance metric selected")
            };
        }

        private void OkButton_Click(object sender, RoutedEventArgs e)
        {
            SelectedAlgorithm = (AlgorithmComboBox.SelectedItem as ComboBoxItem)?.Content.ToString();

            DistanceMetrics = GetSelectedDistanceMetric();


            if (int.TryParse(ClustersTextBox.Text, out var clusters))
                Clusters = clusters;

            //if (double.TryParse(EpsilonTextBox.Text, out var epsilon))
                Epsilon = 0;//epsilon;

            DialogResult = true;
        }

        private void CancelButton_Click(object sender, RoutedEventArgs e)
        {
            DialogResult = false;
        }
    }
}
