using Seq2SeqSharp;
using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Enums;
using Seq2SeqSharp.LearningRate;
using Seq2SeqSharp.Metrics;
using Seq2SeqSharp.Optimizer;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;

namespace ConsoleApplication2Apr2025
{
    internal class Program
    {
        static void Main(string[] args)
        {
            string rootPath = Directory.GetCurrentDirectory();

            string trainFolderPath = Path.Combine(rootPath, "train");
            string validFolderPath = Path.Combine(rootPath, "valid");

            // Ensure training and validation directories exist
            Directory.CreateDirectory(trainFolderPath);
            Directory.CreateDirectory(validFolderPath);

            // List of files to copy
            string[] sourceFiles = { "train.enu.snt", "train.chs.snt" };

            foreach (var fileName in sourceFiles)
            {
                string trainDest = Path.Combine(trainFolderPath, fileName);
                string validDest = Path.Combine(validFolderPath, fileName);

                // Copy training file
                if (!File.Exists(trainDest))
                {
                    File.Copy(fileName, trainDest);
                }

                // Copy validation file
                if (!File.Exists(validDest))
                {
                    File.Copy(fileName, validDest);
                }
            }

            // Build configs for training
            Seq2SeqOptions opts = CreateOptions(trainFolderPath, validFolderPath);

            DecodingOptions decodingOptions = opts.CreateDecodingOptions();

            // Load training corpus
            var trainCorpus = new Seq2SeqCorpus(corpusFilePath: opts.TrainCorpusPath, srcLangName: opts.SrcLang, tgtLangName: opts.TgtLang, maxTokenSizePerBatch: opts.MaxTokenSizePerBatch,
                maxSrcSentLength: opts.MaxSrcSentLength, maxTgtSentLength: opts.MaxTgtSentLength, paddingEnums: opts.PaddingType, tooLongSequence: opts.TooLongSequence);

            // Load valid corpus
            var validCorpusList = new List<Seq2SeqCorpus>();
            if (!opts.ValidCorpusPaths.IsNullOrEmpty())
            {
                string[] validCorpusPathList = opts.ValidCorpusPaths.Split(';');
                foreach (var validCorpusPath in validCorpusPathList)
                {
                    validCorpusList.Add(new Seq2SeqCorpus(validCorpusPath, opts.SrcLang, opts.TgtLang, opts.ValMaxTokenSizePerBatch, opts.MaxValidSrcSentLength, opts.MaxValidTgtSentLength, paddingEnums: opts.PaddingType, tooLongSequence: opts.TooLongSequence));
                }
            }

            // Create learning rate
            ILearningRate learningRate = new DecayLearningRate(opts.StartLearningRate, opts.WarmUpSteps, opts.WeightsUpdateCount, opts.LearningRateStepDownFactor, opts.UpdateNumToStepDownLearningRate);

            // Create optimizer
            IOptimizer optimizer = Misc.CreateOptimizer(opts);

            // Build vocabularies for training
            (var srcVocab, var tgtVocab) = trainCorpus.BuildVocabs(opts.SrcVocabSize, opts.TgtVocabSize, opts.SharedEmbeddings);

            // Create metrics
            List<IMetric> metrics = new List<IMetric> { new BleuMetric() };

            //New training
            var ss = new Seq2Seq(opts, srcVocab, tgtVocab);

            // Add event handler for monitoring       
            ss.StatusUpdateWatcher += Ss_StatusUpdateWatcher;
            ss.EpochEndWatcher += Ss_EpochEndWatcher;

            // Kick off training
            ss.Train(maxTrainingEpoch: opts.MaxEpochNum, trainCorpus: trainCorpus, validCorpusList: validCorpusList.ToArray(), learningRate: learningRate, optimizer: optimizer, metrics: metrics.ToArray(), decodingOptions: decodingOptions);

            ss.SaveModel(suffix: ".test");

            /*
            Update = 8100, Epoch = 0, LR = 0,000596, AvgCost = 4,9785, Sent = 186, SentPerMin = 157,10, WordPerSec = 135,93
            Update = 8200, Epoch = 0, LR = 0,000593, AvgCost = 4,8093, Sent = 380, SentPerMin = 166,14, WordPerSec = 135,98
            Update = 8300, Epoch = 0, LR = 0,000589, AvgCost = 4,7829, Sent = 553, SentPerMin = 165,62, WordPerSec = 135,69
            Update = 8400, Epoch = 0, LR = 0,000586, AvgCost = 4,8824, Sent = 697, SentPerMin = 161,03, WordPerSec = 134,63
            Update = 8500, Epoch = 0, LR = 0,000582, AvgCost = 5,0666, Sent = 821, SentPerMin = 154,83, WordPerSec = 134,21
            Update = 8600, Epoch = 0, LR = 0,000579, AvgCost = 5,2345, Sent = 935, SentPerMin = 149,11, WordPerSec = 133,63
            Update = 8662, Epoch = 0, LR = 0,000577, AvgCost = 5,3364, Sent = 1000, SentPerMin = 145,24, WordPerSec = 133,30
            Update = 8700, Epoch = 1, LR = 0,000575, AvgCost = 3,9885, Sent = 78, SentPerMin = 213,78, WordPerSec = 168,19
            Update = 8800, Epoch = 1, LR = 0,000572, AvgCost = 4,1704, Sent = 277, SentPerMin = 213,25, WordPerSec = 166,04
            Update = 8900, Epoch = 1, LR = 0,000569, AvgCost = 4,4166, Sent = 448, SentPerMin = 200,09, WordPerSec = 166,25
            Update = 9000, Epoch = 1, LR = 0,000566, AvgCost = 4,5607, Sent = 607, SentPerMin = 194,34, WordPerSec = 165,44
            Update = 9100, Epoch = 1, LR = 0,000563, AvgCost = 4,7340, Sent = 742, SentPerMin = 188,23, WordPerSec = 163,49
            Update = 9200, Epoch = 1, LR = 0,000560, AvgCost = 4,9114, Sent = 862, SentPerMin = 180,17, WordPerSec = 161,27
            Update = 9300, Epoch = 1, LR = 0,000556, AvgCost = 5,0460, Sent = 976, SentPerMin = 174,62, WordPerSec = 158,98
            Update = 9324, Epoch = 1, LR = 0,000556, AvgCost = 5,0985, Sent = 1000, SentPerMin = 172,81, WordPerSec = 158,60
            Update = 9400, Epoch = 2, LR = 0,000554, AvgCost = 4,0781, Sent = 151, SentPerMin = 200,25, WordPerSec = 150,46
            Update = 9500, Epoch = 2, LR = 0,000551, AvgCost = 4,2615, Sent = 329, SentPerMin = 187,67, WordPerSec = 146,43
            Update = 9600, Epoch = 2, LR = 0,000548, AvgCost = 4,2781, Sent = 513, SentPerMin = 180,24, WordPerSec = 146,47
            Update = 9700, Epoch = 2, LR = 0,000545, AvgCost = 4,5369, Sent = 647, SentPerMin = 168,32, WordPerSec = 144,52
            Update = 9800, Epoch = 2, LR = 0,000542, AvgCost = 4,7327, Sent = 785, SentPerMin = 162,78, WordPerSec = 143,17
            Update = 9900, Epoch = 2, LR = 0,000539, AvgCost = 4,8857, Sent = 908, SentPerMin = 157,44, WordPerSec = 141,14
            Update = 9986, Epoch = 2, LR = 0,000537, AvgCost = 5,0398, Sent = 1000, SentPerMin = 152,34, WordPerSec = 139,81
            */

            /*
            Update = 8100, Epoch = 0, LR = 0,000596, AvgCost = 4,8622, Sent = 199, SentPerMin = 167,75, WordPerSec = 133,47
            Update = 8200, Epoch = 0, LR = 0,000593, AvgCost = 4,5666, Sent = 402, SentPerMin = 171,89, WordPerSec = 134,79
            Update = 8300, Epoch = 0, LR = 0,000589, AvgCost = 4,8559, Sent = 543, SentPerMin = 160,73, WordPerSec = 134,09
            Update = 8400, Epoch = 0, LR = 0,000586, AvgCost = 5,0071, Sent = 683, SentPerMin = 154,50, WordPerSec = 133,20
            Update = 8500, Epoch = 0, LR = 0,000582, AvgCost = 5,1247, Sent = 813, SentPerMin = 150,15, WordPerSec = 132,45
            Update = 8600, Epoch = 0, LR = 0,000579, AvgCost = 5,2399, Sent = 935, SentPerMin = 146,14, WordPerSec = 131,77
            Update = 8662, Epoch = 0, LR = 0,000577, AvgCost = 5,3413, Sent = 1000, SentPerMin = 143,33, WordPerSec = 131,54
            Update = 8700, Epoch = 1, LR = 0,000575, AvgCost = 3,8443, Sent = 78, SentPerMin = 214,65, WordPerSec = 160,21
            Update = 8800, Epoch = 1, LR = 0,000572, AvgCost = 4,1595, Sent = 276, SentPerMin = 208,79, WordPerSec = 160,82
            Update = 8900, Epoch = 1, LR = 0,000569, AvgCost = 4,5534, Sent = 428, SentPerMin = 193,97, WordPerSec = 158,34
            Update = 9000, Epoch = 1, LR = 0,000566, AvgCost = 4,5705, Sent = 601, SentPerMin = 189,92, WordPerSec = 158,17
            Update = 9100, Epoch = 1, LR = 0,000563, AvgCost = 4,7561, Sent = 734, SentPerMin = 182,27, WordPerSec = 156,97
            Update = 9200, Epoch = 1, LR = 0,000560, AvgCost = 4,9112, Sent = 862, SentPerMin = 174,34, WordPerSec = 156,17
            Update = 9300, Epoch = 1, LR = 0,000556, AvgCost = 5,0648, Sent = 975, SentPerMin = 168,91, WordPerSec = 154,14
            Update = 9324, Epoch = 1, LR = 0,000556, AvgCost = 5,1002, Sent = 1000, SentPerMin = 167,52, WordPerSec = 153,75
            Update = 9400, Epoch = 2, LR = 0,000554, AvgCost = 4,4029, Sent = 138, SentPerMin = 175,66, WordPerSec = 151,05
            Update = 9500, Epoch = 2, LR = 0,000551, AvgCost = 4,2632, Sent = 332, SentPerMin = 179,77, WordPerSec = 148,46
            Update = 9600, Epoch = 2, LR = 0,000548, AvgCost = 4,4767, Sent = 485, SentPerMin = 168,31, WordPerSec = 144,24
            Update = 9700, Epoch = 2, LR = 0,000545, AvgCost = 4,6179, Sent = 648, SentPerMin = 163,36, WordPerSec = 142,30
            Update = 9800, Epoch = 2, LR = 0,000542, AvgCost = 4,7211, Sent = 790, SentPerMin = 158,94, WordPerSec = 139,65
            Update = 9900, Epoch = 2, LR = 0,000539, AvgCost = 4,9137, Sent = 906, SentPerMin = 152,68, WordPerSec = 137,38
            Update = 9986, Epoch = 2, LR = 0,000537, AvgCost = 5,0356, Sent = 1000, SentPerMin = 148,12, WordPerSec = 135,94
            */

            /*
            Update = 8100, Epoch = 0, LR = 0,000596, AvgCost = 4,8502, Sent = 199, SentPerMin = 733,81, WordPerSec = 583,85
            Update = 8200, Epoch = 0, LR = 0,000593, AvgCost = 4,5645, Sent = 402, SentPerMin = 761,53, WordPerSec = 597,20
            Update = 8300, Epoch = 0, LR = 0,000589, AvgCost = 4,8565, Sent = 543, SentPerMin = 700,31, WordPerSec = 584,23
            Update = 8400, Epoch = 0, LR = 0,000586, AvgCost = 5,0067, Sent = 683, SentPerMin = 667,61, WordPerSec = 575,55
            Update = 8500, Epoch = 0, LR = 0,000582, AvgCost = 5,1244, Sent = 813, SentPerMin = 642,56, WordPerSec = 566,81
            Update = 8600, Epoch = 0, LR = 0,000579, AvgCost = 5,2395, Sent = 935, SentPerMin = 620,59, WordPerSec = 559,56
            Update = 8662, Epoch = 0, LR = 0,000577, AvgCost = 5,3404, Sent = 1000, SentPerMin = 604,67, WordPerSec = 554,94
            Update = 8700, Epoch = 1, LR = 0,000575, AvgCost = 3,8426, Sent = 78, SentPerMin = 933,27, WordPerSec = 696,56
            Update = 8800, Epoch = 1, LR = 0,000572, AvgCost = 4,1547, Sent = 276, SentPerMin = 912,24, WordPerSec = 702,63
            Update = 8900, Epoch = 1, LR = 0,000569, AvgCost = 4,5524, Sent = 428, SentPerMin = 833,17, WordPerSec = 680,13
            Update = 9000, Epoch = 1, LR = 0,000566, AvgCost = 4,5690, Sent = 601, SentPerMin = 819,75, WordPerSec = 682,71
            Update = 9100, Epoch = 1, LR = 0,000563, AvgCost = 4,7517, Sent = 734, SentPerMin = 778,88, WordPerSec = 670,77
            Update = 9200, Epoch = 1, LR = 0,000560, AvgCost = 4,9047, Sent = 862, SentPerMin = 744,15, WordPerSec = 666,60
            Update = 9300, Epoch = 1, LR = 0,000556, AvgCost = 5,0580, Sent = 975, SentPerMin = 715,50, WordPerSec = 652,93
            Update = 9324, Epoch = 1, LR = 0,000556, AvgCost = 5,0931, Sent = 1000, SentPerMin = 707,69, WordPerSec = 649,49
            Update = 9400, Epoch = 2, LR = 0,000554, AvgCost = 4,3767, Sent = 138, SentPerMin = 820,11, WordPerSec = 705,21
            Update = 9500, Epoch = 2, LR = 0,000551, AvgCost = 4,2459, Sent = 332, SentPerMin = 857,52, WordPerSec = 708,19
            Update = 9600, Epoch = 2, LR = 0,000548, AvgCost = 4,4566, Sent = 485, SentPerMin = 807,22, WordPerSec = 691,79
            Update = 9700, Epoch = 2, LR = 0,000545, AvgCost = 4,5915, Sent = 648, SentPerMin = 792,93, WordPerSec = 690,72
            Update = 9800, Epoch = 2, LR = 0,000542, AvgCost = 4,6959, Sent = 790, SentPerMin = 764,89, WordPerSec = 672,04
            Update = 9900, Epoch = 2, LR = 0,000539, AvgCost = 4,8862, Sent = 906, SentPerMin = 729,24, WordPerSec = 656,16
            Update = 9986, Epoch = 2, LR = 0,000537, AvgCost = 5,0089, Sent = 1000, SentPerMin = 703,85, WordPerSec = 645,97
            */

            Console.ReadLine();
        }


        private static Seq2SeqOptions CreateOptions(string trainFolderPath, string validFolderPath)
        {
            Seq2SeqOptions opts = new Seq2SeqOptions();
            opts.Task = ModeEnums.Train;

            opts.TrainCorpusPath = trainFolderPath;
            opts.ValidCorpusPaths = validFolderPath;
            opts.SrcLang = "ENU";
            opts.TgtLang = "CHS";

            opts.EncoderLayerDepth = 2;
            opts.DecoderLayerDepth = 2;
            opts.SrcEmbeddingDim = 64;
            opts.TgtEmbeddingDim = 64;
            opts.HiddenSize = 64;
            opts.MultiHeadNum = 8;

            opts.StartLearningRate = 0.0006f;
            opts.WarmUpSteps = 8000;
            opts.WeightsUpdateCount = 8000;

            opts.MaxTokenSizePerBatch = 128;
            opts.ValMaxTokenSizePerBatch = 128;
            opts.MaxSrcSentLength = 110;
            opts.MaxTgtSentLength = 110;
            opts.MaxValidSrcSentLength = 110;
            opts.MaxValidTgtSentLength = 110;
            opts.PaddingType = Seq2SeqSharp.Utils.PaddingEnums.NoPadding;
            opts.TooLongSequence = TooLongSequence.Truncation;
            opts.ProcessorType = ProcessorTypeEnums.CPU;
            opts.MaxEpochNum = 3;

            opts.ModelFilePath = "seq2seq_test.model";

            return opts;
        }


        public static void Ss_StatusUpdateWatcher(object? sender, EventArgs e)
        {
            CostEventArg? ep = e as CostEventArg;
            if (ep != null)
            {
                TimeSpan ts = DateTime.Now - ep.StartDateTime;
                double sentPerMin = 0;
                double wordPerSec = 0;
                if (ts.TotalMinutes > 0)
                {
                    sentPerMin = ep.ProcessedSentencesInTotal / ts.TotalMinutes;
                }

                if (ts.TotalSeconds > 0)
                {
                    wordPerSec = ep.ProcessedWordsInTotal / ts.TotalSeconds;
                }

                Console.WriteLine($"Update = {ep.Update}, Epoch = {ep.Epoch}, LR = {ep.LearningRate:F6}, AvgCost = {ep.AvgCostInTotal:F4}, Sent = {ep.ProcessedSentencesInTotal}, SentPerMin = {sentPerMin:F}, WordPerSec = {wordPerSec:F}");

            }
            else
            {
                throw new ArgumentNullException("The input event argument e is not a CostEventArg.");
            }
        }

        public static void Ss_EpochEndWatcher(object? sender, EventArgs e)
        {
            CostEventArg? ep = e as CostEventArg;

            if (ep != null)
            {
                TimeSpan ts = DateTime.Now - ep.StartDateTime;
                double sentPerMin = 0;
                double wordPerSec = 0;
                if (ts.TotalMinutes > 0)
                {
                    sentPerMin = ep.ProcessedSentencesInTotal / ts.TotalMinutes;
                }

                if (ts.TotalSeconds > 0)
                {
                    wordPerSec = ep.ProcessedWordsInTotal / ts.TotalSeconds;
                }

                Console.WriteLine($"Update = {ep.Update}, Epoch = {ep.Epoch}, LR = {ep.LearningRate:F6}, AvgCost = {ep.AvgCostInTotal:F4}, Sent = {ep.ProcessedSentencesInTotal}, SentPerMin = {sentPerMin:F}, WordPerSec = {wordPerSec:F}");

            }
            else
            {
                throw new ArgumentNullException("The input event argument e is not a CostEventArg.");
            }
        }
    }
}
