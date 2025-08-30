using Seq2SeqSharp;
using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Enums;
using Seq2SeqSharp.LearningRate;
using Seq2SeqSharp.Metrics;
using Seq2SeqSharp.Optimizer;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;

namespace ConsoleApplication5May2025
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;

            string rootPath = Directory.GetCurrentDirectory();

            string trainFolderPath = Path.Combine(rootPath, "train");
            string validFolderPath = Path.Combine(rootPath, "valid");

            Directory.CreateDirectory(trainFolderPath);
            Directory.CreateDirectory(validFolderPath);

            string[] sourceFiles = { "train.enu.snt", "train.chs.snt" };
            CopyTrainingFiles(sourceFiles, trainFolderPath, validFolderPath);

            string[] modelFiles = { "enuSpmZk3k.model", "chsSpmZk4k.model" };
            CopyTrainingFiles(modelFiles, trainFolderPath, validFolderPath);

            string srcSpmPath = Path.Combine(trainFolderPath, "enuSpmZk3k.model");
            string tgtSpmPath = Path.Combine(trainFolderPath, "chsSpmZk4k.model");

            Seq2SeqOptions options = CreateOptions(trainFolderPath, validFolderPath);
            DecodingOptions decodingOptions = options.CreateDecodingOptions();

            var trainCorpus = new Seq2SeqCorpus(
                corpusFilePath: options.TrainCorpusPath,
                srcLangName: options.SrcLang,
                tgtLangName: options.TgtLang,
                maxTokenSizePerBatch: options.MaxTokenSizePerBatch,
                maxSrcSentLength: options.MaxSrcSentLength,
                maxTgtSentLength: options.MaxTgtSentLength,
                paddingEnums: options.PaddingType,
                tooLongSequence: options.TooLongSequence);

            var validCorpusList = new List<Seq2SeqCorpus>();
            if (!options.ValidCorpusPaths.IsNullOrEmpty())
            {
                foreach (var validCorpusPath in options.ValidCorpusPaths.Split(';'))
                {
                    validCorpusList.Add(new Seq2SeqCorpus(
                        validCorpusPath,
                        options.SrcLang,
                        options.TgtLang,
                        options.ValMaxTokenSizePerBatch,
                        options.MaxValidSrcSentLength,
                        options.MaxValidTgtSentLength,
                        paddingEnums: options.PaddingType,
                        tooLongSequence: options.TooLongSequence));
                }
            }

            ILearningRate learningRate = new DecayLearningRate(
                options.StartLearningRate,
                options.WarmUpSteps,
                options.WeightsUpdateCount,
                options.LearningRateStepDownFactor,
                options.UpdateNumToStepDownLearningRate);

            IOptimizer optimizer = Misc.CreateOptimizer(options);
            (var srcVocab, var tgtVocab) = trainCorpus.BuildVocabs(options.SrcVocabSize, options.TgtVocabSize, options.SharedEmbeddings);
            List<IMetric> metrics = new List<IMetric> { new BleuMetric(), new RougeMetric(), new SimilarityMetric() };

            var seq2seq = new Seq2Seq(options, srcVocab, tgtVocab);
            seq2seq.StatusUpdateWatcher += Seq2Seq_StatusUpdateWatcher;
            seq2seq.EpochEndWatcher += Seq2Seq_EpochEndWatcher;

            seq2seq.Train(
                maxTrainingEpoch: options.MaxEpochNum,
                trainCorpus: trainCorpus,
                validCorpusList: validCorpusList.ToArray(),
                learningRate: learningRate,
                optimizer: optimizer,
                metrics: metrics.ToArray(),
                decodingOptions: decodingOptions);

            seq2seq.SaveModel(suffix: ".test");

            Console.WriteLine("Starting inference...");

            options.Task = ModeEnums.Test;
            options.ModelFilePath = "seq2seq_test80.model.test";

            var inferSeq2Seq = new Seq2Seq(options);

            string testInputPath = "test_input.enu.snt";
            string testOutputPath = "test_output.chs.snt";

            File.WriteAllLines(testInputPath, new[]
            {
                "Hello, who are the Italian editors present at the fair?",
                "This is a turning point in the 1980s.",
                "I love discounts like early-bird and all-inclusive offers.",
                "The red carpet was crowded at the airport hangar.",
                "Many wait for them at the military line."
            });

            inferSeq2Seq.Test(
                inputTestFile: testInputPath,
                outputFile: testOutputPath,
                batchSize: 1,
                decodingOptions: decodingOptions,
                srcSpmPath: srcSpmPath,
                tgtSpmPath: tgtSpmPath);

            Console.WriteLine("Inference results:");
            foreach (var line in File.ReadLines(testOutputPath))
            {
                Console.WriteLine(line);
            }


            /*
             Update = 10100, Epoch = 0, LR = 0,000597, AvgCost = 266,3259, Sent = 285, SentPerMin = 98,39, WordPerSec = 92,39
Update = 10200, Epoch = 0, LR = 0,000594, AvgCost = 248,0151, Sent = 592, SentPerMin = 98,92, WordPerSec = 90,74
Update = 10300, Epoch = 0, LR = 0,000591, AvgCost = 245,8498, Sent = 882, SentPerMin = 99,83, WordPerSec = 92,22
Update = 10338, Epoch = 0, LR = 0,000590, AvgCost = 242,6513, Sent = 1000, SentPerMin = 100,71, WordPerSec = 92,43
Update = 10400, Epoch = 1, LR = 0,000588, AvgCost = 232,7473, Sent = 178, SentPerMin = 141,02, WordPerSec = 130,12
Update = 10500, Epoch = 1, LR = 0,000586, AvgCost = 233,7370, Sent = 475, SentPerMin = 140,55, WordPerSec = 129,46
Update = 10600, Epoch = 1, LR = 0,000583, AvgCost = 232,3262, Sent = 771, SentPerMin = 140,29, WordPerSec = 129,67
Update = 10676, Epoch = 1, LR = 0,000581, AvgCost = 229,1170, Sent = 1000, SentPerMin = 142,22, WordPerSec = 130,53
Update = 10700, Epoch = 2, LR = 0,000580, AvgCost = 219,7204, Sent = 76, SentPerMin = 100,77, WordPerSec = 88,88
Update = 10800, Epoch = 2, LR = 0,000577, AvgCost = 230,2802, Sent = 363, SentPerMin = 100,41, WordPerSec = 92,59
Update = 10900, Epoch = 2, LR = 0,000575, AvgCost = 227,1935, Sent = 656, SentPerMin = 101,34, WordPerSec = 93,31
Update = 11000, Epoch = 2, LR = 0,000572, AvgCost = 224,7550, Sent = 961, SentPerMin = 102,13, WordPerSec = 93,61
Update = 11014, Epoch = 2, LR = 0,000572, AvgCost = 225,1256, Sent = 1000, SentPerMin = 102,26, WordPerSec = 93,85
Starting inference...
Inference results:
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,

             */


            /*
             Update = 10100, Epoch = 0, LR = 0,000597, AvgCost = 267,4361, Sent = 285, SentPerMin = 98,33, WordPerSec = 92,33
Update = 10200, Epoch = 0, LR = 0,000594, AvgCost = 248,6926, Sent = 592, SentPerMin = 97,41, WordPerSec = 89,35
Update = 10300, Epoch = 0, LR = 0,000591, AvgCost = 246,2259, Sent = 882, SentPerMin = 98,05, WordPerSec = 90,58
Update = 10338, Epoch = 0, LR = 0,000590, AvgCost = 243,0109, Sent = 1000, SentPerMin = 98,91, WordPerSec = 90,78
Update = 10400, Epoch = 1, LR = 0,000588, AvgCost = 232,3520, Sent = 178, SentPerMin = 98,87, WordPerSec = 91,23
Update = 10500, Epoch = 1, LR = 0,000586, AvgCost = 233,6516, Sent = 475, SentPerMin = 98,74, WordPerSec = 90,95
Update = 10600, Epoch = 1, LR = 0,000583, AvgCost = 232,1179, Sent = 771, SentPerMin = 98,45, WordPerSec = 91,00
Update = 10676, Epoch = 1, LR = 0,000581, AvgCost = 228,9296, Sent = 1000, SentPerMin = 99,76, WordPerSec = 91,55
Update = 10700, Epoch = 2, LR = 0,000580, AvgCost = 219,6280, Sent = 76, SentPerMin = 95,91, WordPerSec = 84,59
Update = 10800, Epoch = 2, LR = 0,000577, AvgCost = 230,4686, Sent = 363, SentPerMin = 96,38, WordPerSec = 88,87
Update = 10900, Epoch = 2, LR = 0,000575, AvgCost = 227,3042, Sent = 656, SentPerMin = 96,89, WordPerSec = 89,21
Update = 11000, Epoch = 2, LR = 0,000572, AvgCost = 224,9098, Sent = 961, SentPerMin = 97,59, WordPerSec = 89,44
Update = 11014, Epoch = 2, LR = 0,000572, AvgCost = 225,2936, Sent = 1000, SentPerMin = 97,05, WordPerSec = 89,07
Update = 11100, Epoch = 3, LR = 0,000569, AvgCost = 228,4792, Sent = 248, SentPerMin = 92,75, WordPerSec = 85,68
Update = 11200, Epoch = 3, LR = 0,000567, AvgCost = 224,7080, Sent = 549, SentPerMin = 91,39, WordPerSec = 83,74
Update = 11300, Epoch = 3, LR = 0,000564, AvgCost = 224,6312, Sent = 841, SentPerMin = 87,96, WordPerSec = 81,00
Update = 11352, Epoch = 3, LR = 0,000563, AvgCost = 223,1748, Sent = 1000, SentPerMin = 88,83, WordPerSec = 81,52
Update = 11400, Epoch = 4, LR = 0,000562, AvgCost = 220,7156, Sent = 140, SentPerMin = 99,78, WordPerSec = 89,56
Update = 11500, Epoch = 4, LR = 0,000560, AvgCost = 228,3259, Sent = 431, SentPerMin = 96,00, WordPerSec = 88,99
Update = 11600, Epoch = 4, LR = 0,000557, AvgCost = 221,1094, Sent = 732, SentPerMin = 97,48, WordPerSec = 89,12
Update = 11690, Epoch = 4, LR = 0,000555, AvgCost = 221,1226, Sent = 1000, SentPerMin = 96,85, WordPerSec = 88,89
Update = 11700, Epoch = 5, LR = 0,000555, AvgCost = 181,3901, Sent = 33, SentPerMin = 91,09, WordPerSec = 70,39
Update = 11800, Epoch = 5, LR = 0,000552, AvgCost = 227,8163, Sent = 317, SentPerMin = 83,07, WordPerSec = 77,35
Update = 11900, Epoch = 5, LR = 0,000550, AvgCost = 221,7990, Sent = 618, SentPerMin = 90,41, WordPerSec = 82,98
Update = 12000, Epoch = 5, LR = 0,000548, AvgCost = 221,2325, Sent = 914, SentPerMin = 93,50, WordPerSec = 86,16
Update = 12028, Epoch = 5, LR = 0,000547, AvgCost = 220,0919, Sent = 1000, SentPerMin = 94,49, WordPerSec = 86,72
Update = 12100, Epoch = 6, LR = 0,000545, AvgCost = 228,2643, Sent = 205, SentPerMin = 99,45, WordPerSec = 92,85
Update = 12200, Epoch = 6, LR = 0,000543, AvgCost = 221,6239, Sent = 506, SentPerMin = 116,68, WordPerSec = 107,02
Update = 12300, Epoch = 6, LR = 0,000541, AvgCost = 221,1400, Sent = 802, SentPerMin = 124,78, WordPerSec = 114,87
Update = 12366, Epoch = 6, LR = 0,000540, AvgCost = 219,1293, Sent = 1000, SentPerMin = 128,56, WordPerSec = 117,99
Update = 12400, Epoch = 7, LR = 0,000539, AvgCost = 209,7231, Sent = 106, SentPerMin = 143,27, WordPerSec = 123,40
Update = 12500, Epoch = 7, LR = 0,000537, AvgCost = 222,7241, Sent = 393, SentPerMin = 120,23, WordPerSec = 110,81
Update = 12600, Epoch = 7, LR = 0,000535, AvgCost = 219,1272, Sent = 689, SentPerMin = 111,73, WordPerSec = 102,42
Update = 12700, Epoch = 7, LR = 0,000532, AvgCost = 218,6041, Sent = 989, SentPerMin = 107,76, WordPerSec = 98,93
Update = 12704, Epoch = 7, LR = 0,000532, AvgCost = 218,4395, Sent = 1000, SentPerMin = 107,74, WordPerSec = 98,88
Update = 12800, Epoch = 8, LR = 0,000530, AvgCost = 229,1457, Sent = 275, SentPerMin = 97,86, WordPerSec = 91,86
Update = 12900, Epoch = 8, LR = 0,000528, AvgCost = 219,0951, Sent = 581, SentPerMin = 98,54, WordPerSec = 90,40
Update = 13000, Epoch = 8, LR = 0,000526, AvgCost = 219,3248, Sent = 870, SentPerMin = 99,16, WordPerSec = 91,55
Update = 13042, Epoch = 8, LR = 0,000525, AvgCost = 217,7008, Sent = 1000, SentPerMin = 99,80, WordPerSec = 91,59
Update = 13100, Epoch = 9, LR = 0,000524, AvgCost = 221,2386, Sent = 166, SentPerMin = 98,96, WordPerSec = 91,19
Update = 13200, Epoch = 9, LR = 0,000522, AvgCost = 221,0533, Sent = 463, SentPerMin = 98,26, WordPerSec = 90,69
Update = 13300, Epoch = 9, LR = 0,000520, AvgCost = 218,4441, Sent = 761, SentPerMin = 98,46, WordPerSec = 90,67
Update = 13380, Epoch = 9, LR = 0,000519, AvgCost = 216,3080, Sent = 1000, SentPerMin = 99,66, WordPerSec = 91,46
Update = 13400, Epoch = 10, LR = 0,000518, AvgCost = 209,7233, Sent = 64, SentPerMin = 102,05, WordPerSec = 88,97
Update = 13500, Epoch = 10, LR = 0,000516, AvgCost = 219,8216, Sent = 352, SentPerMin = 98,35, WordPerSec = 90,44
Update = 13600, Epoch = 10, LR = 0,000514, AvgCost = 217,0893, Sent = 645, SentPerMin = 98,84, WordPerSec = 90,98
Update = 13700, Epoch = 10, LR = 0,000513, AvgCost = 215,9322, Sent = 948, SentPerMin = 99,42, WordPerSec = 91,40
Update = 13718, Epoch = 10, LR = 0,000512, AvgCost = 215,3941, Sent = 1000, SentPerMin = 99,76, WordPerSec = 91,55
Update = 13800, Epoch = 11, LR = 0,000511, AvgCost = 216,8790, Sent = 239, SentPerMin = 100,83, WordPerSec = 92,08
Update = 13900, Epoch = 11, LR = 0,000509, AvgCost = 216,6710, Sent = 536, SentPerMin = 98,88, WordPerSec = 90,89
Update = 14000, Epoch = 11, LR = 0,000507, AvgCost = 214,9874, Sent = 831, SentPerMin = 99,34, WordPerSec = 91,22
Update = 14056, Epoch = 11, LR = 0,000506, AvgCost = 214,1623, Sent = 1000, SentPerMin = 99,71, WordPerSec = 91,51
Update = 14100, Epoch = 12, LR = 0,000505, AvgCost = 212,0973, Sent = 131, SentPerMin = 101,68, WordPerSec = 90,49
Update = 14200, Epoch = 12, LR = 0,000504, AvgCost = 215,8038, Sent = 423, SentPerMin = 97,43, WordPerSec = 89,54
Update = 14300, Epoch = 12, LR = 0,000502, AvgCost = 212,9504, Sent = 720, SentPerMin = 99,17, WordPerSec = 90,76
Update = 14394, Epoch = 12, LR = 0,000500, AvgCost = 213,2999, Sent = 1000, SentPerMin = 99,37, WordPerSec = 91,19
Starting inference...
Inference results:
,,在在
,,在在
,的在
,,的的的的的的的。
,,在

             */


            /*
             Update = 10100, Epoch = 0, LR = 0,000597, AvgCost = 267,8119, Sent = 285, SentPerMin = 98,43, WordPerSec = 92,43
Update = 10200, Epoch = 0, LR = 0,000594, AvgCost = 248,7767, Sent = 592, SentPerMin = 98,78, WordPerSec = 90,61
Update = 10300, Epoch = 0, LR = 0,000591, AvgCost = 247,1348, Sent = 882, SentPerMin = 98,91, WordPerSec = 91,38
Update = 10338, Epoch = 0, LR = 0,000590, AvgCost = 244,1288, Sent = 1000, SentPerMin = 99,65, WordPerSec = 91,46
Update = 10400, Epoch = 1, LR = 0,000588, AvgCost = 232,2611, Sent = 178, SentPerMin = 98,44, WordPerSec = 90,83
Update = 10500, Epoch = 1, LR = 0,000586, AvgCost = 233,7492, Sent = 475, SentPerMin = 98,20, WordPerSec = 90,45
Update = 10600, Epoch = 1, LR = 0,000583, AvgCost = 232,4237, Sent = 771, SentPerMin = 98,04, WordPerSec = 90,62
Update = 10676, Epoch = 1, LR = 0,000581, AvgCost = 229,2276, Sent = 1000, SentPerMin = 99,36, WordPerSec = 91,19
Update = 10700, Epoch = 2, LR = 0,000580, AvgCost = 219,5582, Sent = 76, SentPerMin = 98,63, WordPerSec = 86,99
Update = 10800, Epoch = 2, LR = 0,000577, AvgCost = 230,5052, Sent = 363, SentPerMin = 97,85, WordPerSec = 90,23
Update = 10900, Epoch = 2, LR = 0,000575, AvgCost = 227,3926, Sent = 656, SentPerMin = 98,55, WordPerSec = 90,73
Update = 11000, Epoch = 2, LR = 0,000572, AvgCost = 224,9676, Sent = 961, SentPerMin = 99,30, WordPerSec = 91,02
Update = 11014, Epoch = 2, LR = 0,000572, AvgCost = 225,3393, Sent = 1000, SentPerMin = 99,42, WordPerSec = 91,24
Update = 11100, Epoch = 3, LR = 0,000569, AvgCost = 228,5290, Sent = 248, SentPerMin = 99,17, WordPerSec = 91,62
Update = 11200, Epoch = 3, LR = 0,000567, AvgCost = 224,4024, Sent = 549, SentPerMin = 98,79, WordPerSec = 90,52
Update = 11300, Epoch = 3, LR = 0,000564, AvgCost = 224,5782, Sent = 841, SentPerMin = 98,23, WordPerSec = 90,45
Update = 11352, Epoch = 3, LR = 0,000563, AvgCost = 223,0350, Sent = 1000, SentPerMin = 98,54, WordPerSec = 90,44
Update = 11400, Epoch = 4, LR = 0,000562, AvgCost = 221,2353, Sent = 140, SentPerMin = 99,31, WordPerSec = 89,14
Update = 11500, Epoch = 4, LR = 0,000560, AvgCost = 228,0759, Sent = 431, SentPerMin = 95,92, WordPerSec = 88,91
Update = 11600, Epoch = 4, LR = 0,000557, AvgCost = 221,1915, Sent = 732, SentPerMin = 97,54, WordPerSec = 89,17
Update = 11690, Epoch = 4, LR = 0,000555, AvgCost = 221,3671, Sent = 1000, SentPerMin = 97,30, WordPerSec = 89,30
Update = 11700, Epoch = 5, LR = 0,000555, AvgCost = 183,0963, Sent = 33, SentPerMin = 107,02, WordPerSec = 82,70
Update = 11800, Epoch = 5, LR = 0,000552, AvgCost = 228,0717, Sent = 317, SentPerMin = 95,64, WordPerSec = 89,06
Update = 11900, Epoch = 5, LR = 0,000550, AvgCost = 221,8180, Sent = 618, SentPerMin = 96,25, WordPerSec = 88,34
Update = 12000, Epoch = 5, LR = 0,000548, AvgCost = 221,6764, Sent = 914, SentPerMin = 96,51, WordPerSec = 88,93
Update = 12028, Epoch = 5, LR = 0,000547, AvgCost = 220,5675, Sent = 1000, SentPerMin = 96,98, WordPerSec = 89,01
Update = 12100, Epoch = 6, LR = 0,000545, AvgCost = 228,2196, Sent = 205, SentPerMin = 94,76, WordPerSec = 88,48
Update = 12200, Epoch = 6, LR = 0,000543, AvgCost = 220,9822, Sent = 506, SentPerMin = 96,16, WordPerSec = 88,20
Update = 12300, Epoch = 6, LR = 0,000541, AvgCost = 221,1775, Sent = 802, SentPerMin = 96,80, WordPerSec = 89,12
Update = 12366, Epoch = 6, LR = 0,000540, AvgCost = 219,5391, Sent = 1000, SentPerMin = 97,68, WordPerSec = 89,65
Update = 12400, Epoch = 7, LR = 0,000539, AvgCost = 211,1318, Sent = 106, SentPerMin = 100,66, WordPerSec = 86,70
Update = 12500, Epoch = 7, LR = 0,000537, AvgCost = 222,5048, Sent = 393, SentPerMin = 96,39, WordPerSec = 88,84
Update = 12600, Epoch = 7, LR = 0,000535, AvgCost = 219,2231, Sent = 689, SentPerMin = 98,20, WordPerSec = 90,01
Update = 12700, Epoch = 7, LR = 0,000532, AvgCost = 218,9941, Sent = 989, SentPerMin = 98,12, WordPerSec = 90,08
Update = 12704, Epoch = 7, LR = 0,000532, AvgCost = 218,8493, Sent = 1000, SentPerMin = 98,19, WordPerSec = 90,11
Update = 12800, Epoch = 8, LR = 0,000530, AvgCost = 229,1242, Sent = 275, SentPerMin = 96,21, WordPerSec = 90,31
Update = 12900, Epoch = 8, LR = 0,000528, AvgCost = 218,9621, Sent = 581, SentPerMin = 96,96, WordPerSec = 88,95
Update = 13000, Epoch = 8, LR = 0,000526, AvgCost = 219,3012, Sent = 870, SentPerMin = 97,57, WordPerSec = 90,08
Update = 13042, Epoch = 8, LR = 0,000525, AvgCost = 217,7181, Sent = 1000, SentPerMin = 98,21, WordPerSec = 90,13
Update = 13100, Epoch = 9, LR = 0,000524, AvgCost = 220,3964, Sent = 166, SentPerMin = 97,44, WordPerSec = 89,79
Update = 13200, Epoch = 9, LR = 0,000522, AvgCost = 220,6697, Sent = 463, SentPerMin = 96,64, WordPerSec = 89,20
Update = 13300, Epoch = 9, LR = 0,000520, AvgCost = 218,7127, Sent = 761, SentPerMin = 96,96, WordPerSec = 89,28
Update = 13380, Epoch = 9, LR = 0,000519, AvgCost = 216,5574, Sent = 1000, SentPerMin = 98,14, WordPerSec = 90,07
Update = 13400, Epoch = 10, LR = 0,000518, AvgCost = 210,3238, Sent = 64, SentPerMin = 100,40, WordPerSec = 87,53
Update = 13500, Epoch = 10, LR = 0,000516, AvgCost = 219,6875, Sent = 352, SentPerMin = 96,75, WordPerSec = 88,97
Update = 13600, Epoch = 10, LR = 0,000514, AvgCost = 216,7799, Sent = 645, SentPerMin = 97,31, WordPerSec = 89,57
Update = 13700, Epoch = 10, LR = 0,000513, AvgCost = 215,7004, Sent = 948, SentPerMin = 97,85, WordPerSec = 89,95
Update = 13718, Epoch = 10, LR = 0,000512, AvgCost = 215,1990, Sent = 1000, SentPerMin = 98,19, WordPerSec = 90,11
Update = 13800, Epoch = 11, LR = 0,000511, AvgCost = 217,1851, Sent = 239, SentPerMin = 99,25, WordPerSec = 90,64
Update = 13900, Epoch = 11, LR = 0,000509, AvgCost = 216,3055, Sent = 536, SentPerMin = 97,33, WordPerSec = 89,46
Update = 14000, Epoch = 11, LR = 0,000507, AvgCost = 214,5594, Sent = 831, SentPerMin = 97,76, WordPerSec = 89,76
Update = 14056, Epoch = 11, LR = 0,000506, AvgCost = 214,0183, Sent = 1000, SentPerMin = 98,11, WordPerSec = 90,05
Update = 14100, Epoch = 12, LR = 0,000505, AvgCost = 211,3067, Sent = 131, SentPerMin = 99,94, WordPerSec = 88,94
Update = 14200, Epoch = 12, LR = 0,000504, AvgCost = 215,7797, Sent = 423, SentPerMin = 96,83, WordPerSec = 88,99
Update = 14300, Epoch = 12, LR = 0,000502, AvgCost = 212,6489, Sent = 720, SentPerMin = 98,18, WordPerSec = 89,85
Update = 14394, Epoch = 12, LR = 0,000500, AvgCost = 212,9377, Sent = 1000, SentPerMin = 98,18, WordPerSec = 90,10
Update = 14400, Epoch = 13, LR = 0,000500, AvgCost = 212,6977, Sent = 16, SentPerMin = 97,92, WordPerSec = 95,57
Update = 14500, Epoch = 13, LR = 0,000498, AvgCost = 220,3033, Sent = 304, SentPerMin = 96,92, WordPerSec = 90,52
Update = 14600, Epoch = 13, LR = 0,000497, AvgCost = 213,3385, Sent = 606, SentPerMin = 97,07, WordPerSec = 89,21
Update = 14700, Epoch = 13, LR = 0,000495, AvgCost = 212,9336, Sent = 902, SentPerMin = 97,56, WordPerSec = 89,85
Update = 14732, Epoch = 13, LR = 0,000494, AvgCost = 211,7575, Sent = 1000, SentPerMin = 98,05, WordPerSec = 89,99
Update = 14800, Epoch = 14, LR = 0,000493, AvgCost = 216,8685, Sent = 195, SentPerMin = 97,76, WordPerSec = 89,85
Update = 14900, Epoch = 14, LR = 0,000492, AvgCost = 212,2216, Sent = 495, SentPerMin = 97,52, WordPerSec = 89,13
Update = 15000, Epoch = 14, LR = 0,000490, AvgCost = 213,0695, Sent = 789, SentPerMin = 96,96, WordPerSec = 89,41
Update = 15070, Epoch = 14, LR = 0,000489, AvgCost = 210,8282, Sent = 1000, SentPerMin = 97,94, WordPerSec = 89,88
Update = 15100, Epoch = 15, LR = 0,000488, AvgCost = 208,2172, Sent = 92, SentPerMin = 97,17, WordPerSec = 86,24
Update = 15200, Epoch = 15, LR = 0,000487, AvgCost = 213,5600, Sent = 381, SentPerMin = 95,60, WordPerSec = 87,97
Update = 15300, Epoch = 15, LR = 0,000485, AvgCost = 210,9118, Sent = 676, SentPerMin = 96,23, WordPerSec = 88,34
Update = 15400, Epoch = 15, LR = 0,000483, AvgCost = 208,9590, Sent = 979, SentPerMin = 96,81, WordPerSec = 88,65
Update = 15408, Epoch = 15, LR = 0,000483, AvgCost = 210,0689, Sent = 1000, SentPerMin = 96,69, WordPerSec = 88,74
Update = 15500, Epoch = 16, LR = 0,000482, AvgCost = 216,4849, Sent = 265, SentPerMin = 95,79, WordPerSec = 88,90
Update = 15600, Epoch = 16, LR = 0,000480, AvgCost = 210,9519, Sent = 566, SentPerMin = 96,38, WordPerSec = 88,71
Update = 15700, Epoch = 16, LR = 0,000479, AvgCost = 210,5898, Sent = 859, SentPerMin = 96,93, WordPerSec = 89,32
Update = 15746, Epoch = 16, LR = 0,000478, AvgCost = 209,3624, Sent = 1000, SentPerMin = 97,69, WordPerSec = 89,66
Update = 15800, Epoch = 17, LR = 0,000477, AvgCost = 210,6843, Sent = 157, SentPerMin = 97,95, WordPerSec = 89,11
Update = 15900, Epoch = 17, LR = 0,000476, AvgCost = 212,2165, Sent = 450, SentPerMin = 97,09, WordPerSec = 89,52
Update = 16000, Epoch = 17, LR = 0,000474, AvgCost = 207,8301, Sent = 752, SentPerMin = 97,34, WordPerSec = 89,24
Update = 16084, Epoch = 17, LR = 0,000473, AvgCost = 207,9990, Sent = 1000, SentPerMin = 98,07, WordPerSec = 90,00
Update = 16100, Epoch = 18, LR = 0,000473, AvgCost = 201,9287, Sent = 51, SentPerMin = 102,17, WordPerSec = 88,08
Update = 16200, Epoch = 18, LR = 0,000471, AvgCost = 213,1335, Sent = 336, SentPerMin = 97,10, WordPerSec = 89,97
Update = 16300, Epoch = 18, LR = 0,000470, AvgCost = 207,3456, Sent = 636, SentPerMin = 97,43, WordPerSec = 89,26
Update = 16400, Epoch = 18, LR = 0,000469, AvgCost = 207,4117, Sent = 935, SentPerMin = 97,65, WordPerSec = 89,77
Update = 16422, Epoch = 18, LR = 0,000468, AvgCost = 206,9460, Sent = 1000, SentPerMin = 98,12, WordPerSec = 90,05
Update = 16500, Epoch = 19, LR = 0,000467, AvgCost = 209,9110, Sent = 225, SentPerMin = 98,60, WordPerSec = 90,64
Update = 16600, Epoch = 19, LR = 0,000466, AvgCost = 208,8857, Sent = 524, SentPerMin = 97,17, WordPerSec = 89,41
Update = 16700, Epoch = 19, LR = 0,000464, AvgCost = 207,3364, Sent = 819, SentPerMin = 97,66, WordPerSec = 89,83
Update = 16760, Epoch = 19, LR = 0,000463, AvgCost = 206,2634, Sent = 1000, SentPerMin = 98,23, WordPerSec = 90,15
Update = 16800, Epoch = 20, LR = 0,000463, AvgCost = 202,2307, Sent = 122, SentPerMin = 100,95, WordPerSec = 88,11
Update = 16900, Epoch = 20, LR = 0,000462, AvgCost = 209,0176, Sent = 411, SentPerMin = 96,47, WordPerSec = 88,85
Update = 17000, Epoch = 20, LR = 0,000460, AvgCost = 205,2886, Sent = 709, SentPerMin = 98,39, WordPerSec = 89,88
Update = 17098, Epoch = 20, LR = 0,000459, AvgCost = 205,7438, Sent = 1000, SentPerMin = 98,10, WordPerSec = 90,03
Update = 17100, Epoch = 21, LR = 0,000459, AvgCost = 183,3528, Sent = 6, SentPerMin = 112,43, WordPerSec = 102,12
Update = 17200, Epoch = 21, LR = 0,000457, AvgCost = 214,3266, Sent = 292, SentPerMin = 96,59, WordPerSec = 90,37
Update = 17300, Epoch = 21, LR = 0,000456, AvgCost = 207,2260, Sent = 595, SentPerMin = 96,91, WordPerSec = 89,20
Update = 17400, Epoch = 21, LR = 0,000455, AvgCost = 206,2462, Sent = 890, SentPerMin = 97,61, WordPerSec = 89,98
Update = 17436, Epoch = 21, LR = 0,000454, AvgCost = 204,9236, Sent = 1000, SentPerMin = 98,11, WordPerSec = 90,04
Update = 17500, Epoch = 22, LR = 0,000454, AvgCost = 211,7387, Sent = 184, SentPerMin = 98,05, WordPerSec = 90,08
Update = 17600, Epoch = 22, LR = 0,000452, AvgCost = 207,1609, Sent = 482, SentPerMin = 97,20, WordPerSec = 89,36
Update = 17700, Epoch = 22, LR = 0,000451, AvgCost = 207,3049, Sent = 775, SentPerMin = 96,68, WordPerSec = 89,57
Update = 17774, Epoch = 22, LR = 0,000450, AvgCost = 204,1287, Sent = 1000, SentPerMin = 98,14, WordPerSec = 90,07
Update = 17800, Epoch = 23, LR = 0,000450, AvgCost = 199,1985, Sent = 81, SentPerMin = 97,73, WordPerSec = 86,75
Update = 17900, Epoch = 23, LR = 0,000448, AvgCost = 208,5525, Sent = 368, SentPerMin = 96,18, WordPerSec = 89,01
Update = 18000, Epoch = 23, LR = 0,000447, AvgCost = 204,8215, Sent = 663, SentPerMin = 97,56, WordPerSec = 89,63
Update = 18100, Epoch = 23, LR = 0,000446, AvgCost = 202,4237, Sent = 969, SentPerMin = 98,31, WordPerSec = 89,90
Update = 18112, Epoch = 23, LR = 0,000446, AvgCost = 203,3074, Sent = 1000, SentPerMin = 98,17, WordPerSec = 90,10
Update = 18200, Epoch = 24, LR = 0,000445, AvgCost = 211,9268, Sent = 251, SentPerMin = 97,11, WordPerSec = 90,72
Update = 18300, Epoch = 24, LR = 0,000444, AvgCost = 203,4159, Sent = 556, SentPerMin = 97,76, WordPerSec = 89,48
Update = 18400, Epoch = 24, LR = 0,000442, AvgCost = 203,7625, Sent = 847, SentPerMin = 97,45, WordPerSec = 89,82
Update = 18450, Epoch = 24, LR = 0,000442, AvgCost = 202,4331, Sent = 1000, SentPerMin = 98,05, WordPerSec = 89,99
Update = 18500, Epoch = 25, LR = 0,000441, AvgCost = 204,2395, Sent = 145, SentPerMin = 97,78, WordPerSec = 88,93
Update = 18600, Epoch = 25, LR = 0,000440, AvgCost = 207,0203, Sent = 437, SentPerMin = 96,62, WordPerSec = 89,53
Update = 18700, Epoch = 25, LR = 0,000439, AvgCost = 201,5781, Sent = 739, SentPerMin = 97,43, WordPerSec = 89,31
Update = 18788, Epoch = 25, LR = 0,000438, AvgCost = 201,6009, Sent = 1000, SentPerMin = 98,11, WordPerSec = 90,04
Starting inference...
Inference results:
••• ( ( (
和的的在    的) 和  和和。
和
和"•• ( (  )) 的的在• ( •  ) )
"和的•   (  )) 和的在和在


             */


            /*
             Update = 10100, Epoch = 0, LR = 0,000597, AvgCost = 267,4044, Sent = 285, SentPerMin = 98,99, WordPerSec = 92,95
Update = 10200, Epoch = 0, LR = 0,000594, AvgCost = 248,4629, Sent = 592, SentPerMin = 100,18, WordPerSec = 91,90
Update = 10300, Epoch = 0, LR = 0,000591, AvgCost = 246,5326, Sent = 882, SentPerMin = 100,59, WordPerSec = 92,93
Update = 10338, Epoch = 0, LR = 0,000590, AvgCost = 243,3682, Sent = 1000, SentPerMin = 101,42, WordPerSec = 93,08
Update = 10400, Epoch = 1, LR = 0,000588, AvgCost = 233,3631, Sent = 178, SentPerMin = 139,53, WordPerSec = 128,75
Update = 10500, Epoch = 1, LR = 0,000586, AvgCost = 234,1593, Sent = 475, SentPerMin = 139,47, WordPerSec = 128,47
Update = 10600, Epoch = 1, LR = 0,000583, AvgCost = 232,7714, Sent = 771, SentPerMin = 139,64, WordPerSec = 129,07
Update = 10676, Epoch = 1, LR = 0,000581, AvgCost = 229,5322, Sent = 1000, SentPerMin = 141,84, WordPerSec = 130,18
Update = 10700, Epoch = 2, LR = 0,000580, AvgCost = 219,9689, Sent = 76, SentPerMin = 101,26, WordPerSec = 89,32
Update = 10800, Epoch = 2, LR = 0,000577, AvgCost = 230,6265, Sent = 363, SentPerMin = 100,48, WordPerSec = 92,65
Update = 10900, Epoch = 2, LR = 0,000575, AvgCost = 227,5706, Sent = 656, SentPerMin = 101,12, WordPerSec = 93,10
Update = 11000, Epoch = 2, LR = 0,000572, AvgCost = 225,0400, Sent = 961, SentPerMin = 101,82, WordPerSec = 93,32
Update = 11014, Epoch = 2, LR = 0,000572, AvgCost = 225,3956, Sent = 1000, SentPerMin = 101,93, WordPerSec = 93,55
Update = 11100, Epoch = 3, LR = 0,000569, AvgCost = 228,2015, Sent = 248, SentPerMin = 141,06, WordPerSec = 130,31
Update = 11200, Epoch = 3, LR = 0,000567, AvgCost = 224,6363, Sent = 549, SentPerMin = 140,52, WordPerSec = 128,75
Update = 11300, Epoch = 3, LR = 0,000564, AvgCost = 224,7562, Sent = 841, SentPerMin = 140,88, WordPerSec = 129,73
Update = 11352, Epoch = 3, LR = 0,000563, AvgCost = 223,2684, Sent = 1000, SentPerMin = 141,90, WordPerSec = 130,23
Update = 11400, Epoch = 4, LR = 0,000562, AvgCost = 220,3914, Sent = 140, SentPerMin = 103,46, WordPerSec = 92,87
Update = 11500, Epoch = 4, LR = 0,000560, AvgCost = 228,1959, Sent = 431, SentPerMin = 100,13, WordPerSec = 92,82
Update = 11600, Epoch = 4, LR = 0,000557, AvgCost = 221,2932, Sent = 732, SentPerMin = 101,83, WordPerSec = 93,10
Update = 11690, Epoch = 4, LR = 0,000555, AvgCost = 221,5501, Sent = 1000, SentPerMin = 101,81, WordPerSec = 93,44
Update = 11700, Epoch = 5, LR = 0,000555, AvgCost = 183,1615, Sent = 33, SentPerMin = 155,63, WordPerSec = 120,26
Update = 11800, Epoch = 5, LR = 0,000552, AvgCost = 227,5501, Sent = 317, SentPerMin = 139,19, WordPerSec = 129,62
Update = 11900, Epoch = 5, LR = 0,000550, AvgCost = 221,3653, Sent = 618, SentPerMin = 140,20, WordPerSec = 128,67
Update = 12000, Epoch = 5, LR = 0,000548, AvgCost = 220,9329, Sent = 914, SentPerMin = 140,82, WordPerSec = 129,77
Update = 12028, Epoch = 5, LR = 0,000547, AvgCost = 219,7902, Sent = 1000, SentPerMin = 141,75, WordPerSec = 130,10
Update = 12100, Epoch = 6, LR = 0,000545, AvgCost = 227,3135, Sent = 205, SentPerMin = 100,43, WordPerSec = 93,76
Update = 12200, Epoch = 6, LR = 0,000543, AvgCost = 220,4105, Sent = 506, SentPerMin = 100,79, WordPerSec = 92,44
Update = 12300, Epoch = 6, LR = 0,000541, AvgCost = 220,5710, Sent = 802, SentPerMin = 101,12, WordPerSec = 93,09
Update = 12366, Epoch = 6, LR = 0,000540, AvgCost = 218,7415, Sent = 1000, SentPerMin = 101,91, WordPerSec = 93,53
Update = 12400, Epoch = 7, LR = 0,000539, AvgCost = 208,2737, Sent = 106, SentPerMin = 143,66, WordPerSec = 123,73
Update = 12500, Epoch = 7, LR = 0,000537, AvgCost = 220,9240, Sent = 393, SentPerMin = 138,38, WordPerSec = 127,54
Update = 12600, Epoch = 7, LR = 0,000535, AvgCost = 217,7857, Sent = 689, SentPerMin = 141,68, WordPerSec = 129,88
Update = 12700, Epoch = 7, LR = 0,000532, AvgCost = 217,5352, Sent = 989, SentPerMin = 141,54, WordPerSec = 129,94
Update = 12704, Epoch = 7, LR = 0,000532, AvgCost = 217,4009, Sent = 1000, SentPerMin = 141,65, WordPerSec = 130,00
Update = 12800, Epoch = 8, LR = 0,000530, AvgCost = 227,5777, Sent = 275, SentPerMin = 99,84, WordPerSec = 93,72
Update = 12900, Epoch = 8, LR = 0,000528, AvgCost = 217,7925, Sent = 581, SentPerMin = 100,51, WordPerSec = 92,20
Update = 13000, Epoch = 8, LR = 0,000526, AvgCost = 218,4438, Sent = 870, SentPerMin = 101,12, WordPerSec = 93,36
Update = 13042, Epoch = 8, LR = 0,000525, AvgCost = 216,9569, Sent = 1000, SentPerMin = 101,73, WordPerSec = 93,37
Update = 13100, Epoch = 9, LR = 0,000524, AvgCost = 219,9351, Sent = 166, SentPerMin = 139,44, WordPerSec = 128,49
Update = 13200, Epoch = 9, LR = 0,000522, AvgCost = 219,5765, Sent = 463, SentPerMin = 138,64, WordPerSec = 127,96
Update = 13300, Epoch = 9, LR = 0,000520, AvgCost = 217,6476, Sent = 761, SentPerMin = 139,33, WordPerSec = 128,30
Update = 13380, Epoch = 9, LR = 0,000519, AvgCost = 215,7271, Sent = 1000, SentPerMin = 141,39, WordPerSec = 129,76
Update = 13400, Epoch = 10, LR = 0,000518, AvgCost = 211,5836, Sent = 64, SentPerMin = 103,84, WordPerSec = 90,53
Update = 13500, Epoch = 10, LR = 0,000516, AvgCost = 219,6026, Sent = 352, SentPerMin = 100,31, WordPerSec = 92,24
Update = 13600, Epoch = 10, LR = 0,000514, AvgCost = 216,6109, Sent = 645, SentPerMin = 100,82, WordPerSec = 92,80
Update = 13700, Epoch = 10, LR = 0,000513, AvgCost = 215,5468, Sent = 948, SentPerMin = 101,38, WordPerSec = 93,20
Update = 13718, Epoch = 10, LR = 0,000512, AvgCost = 215,0852, Sent = 1000, SentPerMin = 101,72, WordPerSec = 93,35
Update = 13800, Epoch = 11, LR = 0,000511, AvgCost = 216,8411, Sent = 239, SentPerMin = 142,58, WordPerSec = 130,21
Update = 13900, Epoch = 11, LR = 0,000509, AvgCost = 216,3656, Sent = 536, SentPerMin = 139,34, WordPerSec = 128,08
Update = 14000, Epoch = 11, LR = 0,000507, AvgCost = 214,7119, Sent = 831, SentPerMin = 140,48, WordPerSec = 128,99
Update = 14056, Epoch = 11, LR = 0,000506, AvgCost = 213,9475, Sent = 1000, SentPerMin = 141,29, WordPerSec = 129,67
Update = 14100, Epoch = 12, LR = 0,000505, AvgCost = 211,4735, Sent = 131, SentPerMin = 103,53, WordPerSec = 92,13
Update = 14200, Epoch = 12, LR = 0,000504, AvgCost = 216,1875, Sent = 423, SentPerMin = 100,42, WordPerSec = 92,28
Update = 14300, Epoch = 12, LR = 0,000502, AvgCost = 213,2843, Sent = 720, SentPerMin = 101,79, WordPerSec = 93,15
Update = 14394, Epoch = 12, LR = 0,000500, AvgCost = 213,3447, Sent = 1000, SentPerMin = 101,80, WordPerSec = 93,43
Update = 14400, Epoch = 13, LR = 0,000500, AvgCost = 212,3239, Sent = 16, SentPerMin = 138,70, WordPerSec = 135,37
Update = 14500, Epoch = 13, LR = 0,000498, AvgCost = 220,9701, Sent = 304, SentPerMin = 139,77, WordPerSec = 130,54
Update = 14600, Epoch = 13, LR = 0,000497, AvgCost = 213,9900, Sent = 606, SentPerMin = 139,82, WordPerSec = 128,50
Update = 14700, Epoch = 13, LR = 0,000495, AvgCost = 213,5074, Sent = 902, SentPerMin = 140,58, WordPerSec = 129,47
Update = 14732, Epoch = 13, LR = 0,000494, AvgCost = 212,3411, Sent = 1000, SentPerMin = 141,50, WordPerSec = 129,86
Update = 14800, Epoch = 14, LR = 0,000493, AvgCost = 217,9636, Sent = 195, SentPerMin = 101,63, WordPerSec = 93,42
Update = 14900, Epoch = 14, LR = 0,000492, AvgCost = 213,7284, Sent = 495, SentPerMin = 101,46, WordPerSec = 92,72
Update = 15000, Epoch = 14, LR = 0,000490, AvgCost = 214,4117, Sent = 789, SentPerMin = 100,78, WordPerSec = 92,92
Update = 15070, Epoch = 14, LR = 0,000489, AvgCost = 212,0555, Sent = 1000, SentPerMin = 101,79, WordPerSec = 93,42
Update = 15100, Epoch = 15, LR = 0,000488, AvgCost = 209,7300, Sent = 92, SentPerMin = 139,48, WordPerSec = 123,79
Update = 15200, Epoch = 15, LR = 0,000487, AvgCost = 214,4137, Sent = 381, SentPerMin = 138,75, WordPerSec = 127,68
Update = 15300, Epoch = 15, LR = 0,000485, AvgCost = 211,9849, Sent = 676, SentPerMin = 140,91, WordPerSec = 129,37
Update = 15400, Epoch = 15, LR = 0,000483, AvgCost = 209,6278, Sent = 979, SentPerMin = 141,82, WordPerSec = 129,86
Update = 15408, Epoch = 15, LR = 0,000483, AvgCost = 210,7053, Sent = 1000, SentPerMin = 141,64, WordPerSec = 129,99
Update = 15500, Epoch = 16, LR = 0,000482, AvgCost = 217,8386, Sent = 265, SentPerMin = 101,16, WordPerSec = 93,88
Update = 15600, Epoch = 16, LR = 0,000480, AvgCost = 212,5238, Sent = 566, SentPerMin = 100,81, WordPerSec = 92,79
Update = 15700, Epoch = 16, LR = 0,000479, AvgCost = 211,6693, Sent = 859, SentPerMin = 101,04, WordPerSec = 93,11
Update = 15746, Epoch = 16, LR = 0,000478, AvgCost = 210,3247, Sent = 1000, SentPerMin = 101,74, WordPerSec = 93,38
Update = 15800, Epoch = 17, LR = 0,000477, AvgCost = 212,0587, Sent = 157, SentPerMin = 140,37, WordPerSec = 127,69
Update = 15900, Epoch = 17, LR = 0,000476, AvgCost = 213,7671, Sent = 450, SentPerMin = 139,45, WordPerSec = 128,59
Update = 16000, Epoch = 17, LR = 0,000474, AvgCost = 209,1982, Sent = 752, SentPerMin = 140,28, WordPerSec = 128,61
Update = 16084, Epoch = 17, LR = 0,000473, AvgCost = 209,1490, Sent = 1000, SentPerMin = 141,72, WordPerSec = 130,07
Update = 16100, Epoch = 18, LR = 0,000473, AvgCost = 202,9216, Sent = 51, SentPerMin = 106,05, WordPerSec = 91,43
Update = 16200, Epoch = 18, LR = 0,000471, AvgCost = 214,9017, Sent = 336, SentPerMin = 100,71, WordPerSec = 93,31
Update = 16300, Epoch = 18, LR = 0,000470, AvgCost = 209,6384, Sent = 636, SentPerMin = 101,03, WordPerSec = 92,56
Update = 16400, Epoch = 18, LR = 0,000469, AvgCost = 209,4314, Sent = 935, SentPerMin = 101,25, WordPerSec = 93,07
Update = 16422, Epoch = 18, LR = 0,000468, AvgCost = 209,0505, Sent = 1000, SentPerMin = 101,76, WordPerSec = 93,39
Update = 16500, Epoch = 19, LR = 0,000467, AvgCost = 211,1321, Sent = 225, SentPerMin = 142,02, WordPerSec = 130,55
Update = 16600, Epoch = 19, LR = 0,000466, AvgCost = 210,3219, Sent = 524, SentPerMin = 139,36, WordPerSec = 128,23
Update = 16700, Epoch = 19, LR = 0,000464, AvgCost = 208,6958, Sent = 819, SentPerMin = 140,35, WordPerSec = 129,10
Update = 16760, Epoch = 19, LR = 0,000463, AvgCost = 207,5769, Sent = 1000, SentPerMin = 141,46, WordPerSec = 129,83
Update = 16800, Epoch = 20, LR = 0,000463, AvgCost = 204,2459, Sent = 122, SentPerMin = 104,76, WordPerSec = 91,44
Update = 16900, Epoch = 20, LR = 0,000462, AvgCost = 210,0371, Sent = 411, SentPerMin = 100,10, WordPerSec = 92,19
Update = 17000, Epoch = 20, LR = 0,000460, AvgCost = 206,5661, Sent = 709, SentPerMin = 102,05, WordPerSec = 93,22
Update = 17098, Epoch = 20, LR = 0,000459, AvgCost = 207,2004, Sent = 1000, SentPerMin = 101,70, WordPerSec = 93,34
Update = 17100, Epoch = 21, LR = 0,000459, AvgCost = 183,2323, Sent = 6, SentPerMin = 155,06, WordPerSec = 140,85
Update = 17200, Epoch = 21, LR = 0,000457, AvgCost = 214,3915, Sent = 292, SentPerMin = 138,97, WordPerSec = 130,02
Update = 17300, Epoch = 21, LR = 0,000456, AvgCost = 207,6437, Sent = 595, SentPerMin = 139,24, WordPerSec = 128,16
Update = 17400, Epoch = 21, LR = 0,000455, AvgCost = 207,0306, Sent = 890, SentPerMin = 140,33, WordPerSec = 129,36
Update = 17436, Epoch = 21, LR = 0,000454, AvgCost = 205,8346, Sent = 1000, SentPerMin = 141,34, WordPerSec = 129,72
Update = 17500, Epoch = 22, LR = 0,000454, AvgCost = 211,9407, Sent = 184, SentPerMin = 101,35, WordPerSec = 93,11
Update = 17600, Epoch = 22, LR = 0,000452, AvgCost = 208,4228, Sent = 482, SentPerMin = 100,63, WordPerSec = 92,52
Update = 17700, Epoch = 22, LR = 0,000451, AvgCost = 209,0709, Sent = 775, SentPerMin = 100,11, WordPerSec = 92,76
Update = 17774, Epoch = 22, LR = 0,000450, AvgCost = 206,0451, Sent = 1000, SentPerMin = 101,60, WordPerSec = 93,24
Update = 17800, Epoch = 23, LR = 0,000450, AvgCost = 200,6660, Sent = 81, SentPerMin = 139,40, WordPerSec = 123,74
Update = 17900, Epoch = 23, LR = 0,000448, AvgCost = 209,3797, Sent = 368, SentPerMin = 137,62, WordPerSec = 127,36
Update = 18000, Epoch = 23, LR = 0,000447, AvgCost = 205,6865, Sent = 663, SentPerMin = 140,27, WordPerSec = 128,88
Update = 18100, Epoch = 23, LR = 0,000446, AvgCost = 203,5669, Sent = 969, SentPerMin = 141,34, WordPerSec = 129,25
Update = 18112, Epoch = 23, LR = 0,000446, AvgCost = 204,5228, Sent = 1000, SentPerMin = 141,28, WordPerSec = 129,66
Update = 18200, Epoch = 24, LR = 0,000445, AvgCost = 213,0698, Sent = 251, SentPerMin = 100,68, WordPerSec = 94,05
Update = 18300, Epoch = 24, LR = 0,000444, AvgCost = 204,6508, Sent = 556, SentPerMin = 101,28, WordPerSec = 92,71
Update = 18400, Epoch = 24, LR = 0,000442, AvgCost = 205,3650, Sent = 847, SentPerMin = 101,03, WordPerSec = 93,12
Update = 18450, Epoch = 24, LR = 0,000442, AvgCost = 204,2869, Sent = 1000, SentPerMin = 101,64, WordPerSec = 93,28
Update = 18500, Epoch = 25, LR = 0,000441, AvgCost = 205,0191, Sent = 145, SentPerMin = 139,21, WordPerSec = 126,62
Update = 18600, Epoch = 25, LR = 0,000440, AvgCost = 208,0286, Sent = 437, SentPerMin = 138,40, WordPerSec = 128,25
Update = 18700, Epoch = 25, LR = 0,000439, AvgCost = 202,6341, Sent = 739, SentPerMin = 139,78, WordPerSec = 128,14
Update = 18788, Epoch = 25, LR = 0,000438, AvgCost = 202,8645, Sent = 1000, SentPerMin = 141,07, WordPerSec = 129,47
Update = 18800, Epoch = 26, LR = 0,000438, AvgCost = 164,2800, Sent = 41, SentPerMin = 110,95, WordPerSec = 85,42
Update = 18900, Epoch = 26, LR = 0,000436, AvgCost = 208,8509, Sent = 324, SentPerMin = 100,23, WordPerSec = 93,10
Update = 19000, Epoch = 26, LR = 0,000435, AvgCost = 203,7117, Sent = 624, SentPerMin = 100,73, WordPerSec = 92,35
Update = 19100, Epoch = 26, LR = 0,000434, AvgCost = 203,4348, Sent = 921, SentPerMin = 101,21, WordPerSec = 93,12
Update = 19126, Epoch = 26, LR = 0,000434, AvgCost = 202,8020, Sent = 1000, SentPerMin = 101,51, WordPerSec = 93,16
Update = 19200, Epoch = 27, LR = 0,000433, AvgCost = 206,3596, Sent = 213, SentPerMin = 140,26, WordPerSec = 129,39
Update = 19300, Epoch = 27, LR = 0,000432, AvgCost = 204,4928, Sent = 511, SentPerMin = 138,05, WordPerSec = 127,29
Update = 19400, Epoch = 27, LR = 0,000431, AvgCost = 202,9886, Sent = 808, SentPerMin = 139,87, WordPerSec = 128,84
Update = 19464, Epoch = 27, LR = 0,000430, AvgCost = 201,4430, Sent = 1000, SentPerMin = 141,13, WordPerSec = 129,52
Update = 19500, Epoch = 28, LR = 0,000430, AvgCost = 190,5038, Sent = 113, SentPerMin = 105,12, WordPerSec = 90,05
Update = 19600, Epoch = 28, LR = 0,000429, AvgCost = 203,9936, Sent = 399, SentPerMin = 100,05, WordPerSec = 92,10
Update = 19700, Epoch = 28, LR = 0,000427, AvgCost = 201,7038, Sent = 696, SentPerMin = 101,45, WordPerSec = 92,95
Update = 19800, Epoch = 28, LR = 0,000426, AvgCost = 201,3920, Sent = 995, SentPerMin = 101,62, WordPerSec = 93,20
Update = 19802, Epoch = 28, LR = 0,000426, AvgCost = 201,3526, Sent = 1000, SentPerMin = 101,59, WordPerSec = 93,23
Update = 19900, Epoch = 29, LR = 0,000425, AvgCost = 209,9653, Sent = 280, SentPerMin = 138,43, WordPerSec = 129,90
Update = 20000, Epoch = 29, LR = 0,000424, AvgCost = 200,9273, Sent = 588, SentPerMin = 139,49, WordPerSec = 127,60
Update = 20100, Epoch = 29, LR = 0,000423, AvgCost = 201,8063, Sent = 877, SentPerMin = 89,47, WordPerSec = 82,52
Update = 20140, Epoch = 29, LR = 0,000423, AvgCost = 200,6789, Sent = 1000, SentPerMin = 91,25, WordPerSec = 83,75
Update = 20200, Epoch = 30, LR = 0,000422, AvgCost = 203,0004, Sent = 173, SentPerMin = 139,07, WordPerSec = 127,95
Update = 20300, Epoch = 30, LR = 0,000421, AvgCost = 204,5678, Sent = 469, SentPerMin = 138,90, WordPerSec = 128,13
Update = 20400, Epoch = 30, LR = 0,000420, AvgCost = 202,9972, Sent = 766, SentPerMin = 139,12, WordPerSec = 128,38
Update = 20478, Epoch = 30, LR = 0,000419, AvgCost = 200,6457, Sent = 1000, SentPerMin = 141,52, WordPerSec = 129,88
Update = 20500, Epoch = 31, LR = 0,000419, AvgCost = 192,2429, Sent = 71, SentPerMin = 104,39, WordPerSec = 90,35
Update = 20600, Epoch = 31, LR = 0,000418, AvgCost = 202,8769, Sent = 358, SentPerMin = 101,17, WordPerSec = 92,93
Update = 20700, Epoch = 31, LR = 0,000417, AvgCost = 200,7286, Sent = 652, SentPerMin = 102,21, WordPerSec = 93,85
Update = 20800, Epoch = 31, LR = 0,000416, AvgCost = 199,7184, Sent = 955, SentPerMin = 102,56, WordPerSec = 94,09
Update = 20816, Epoch = 31, LR = 0,000416, AvgCost = 199,7717, Sent = 1000, SentPerMin = 102,81, WordPerSec = 94,35
Update = 20900, Epoch = 32, LR = 0,000415, AvgCost = 199,8366, Sent = 244, SentPerMin = 147,82, WordPerSec = 135,05
Update = 21000, Epoch = 32, LR = 0,000414, AvgCost = 199,9973, Sent = 542, SentPerMin = 144,33, WordPerSec = 132,56
Update = 21100, Epoch = 32, LR = 0,000413, AvgCost = 199,6315, Sent = 835, SentPerMin = 144,42, WordPerSec = 132,87
Update = 21154, Epoch = 32, LR = 0,000413, AvgCost = 198,6789, Sent = 1000, SentPerMin = 145,31, WordPerSec = 133,36
Update = 21200, Epoch = 33, LR = 0,000412, AvgCost = 196,3095, Sent = 136, SentPerMin = 107,29, WordPerSec = 95,47
Update = 21300, Epoch = 33, LR = 0,000411, AvgCost = 201,2504, Sent = 428, SentPerMin = 103,77, WordPerSec = 95,44
Update = 21400, Epoch = 33, LR = 0,000410, AvgCost = 198,8943, Sent = 725, SentPerMin = 105,12, WordPerSec = 96,36
Update = 21492, Epoch = 33, LR = 0,000409, AvgCost = 198,9201, Sent = 1000, SentPerMin = 105,28, WordPerSec = 96,62
Update = 21500, Epoch = 34, LR = 0,000409, AvgCost = 181,3519, Sent = 23, SentPerMin = 146,24, WordPerSec = 127,48
Update = 21600, Epoch = 34, LR = 0,000408, AvgCost = 204,1405, Sent = 311, SentPerMin = 144,36, WordPerSec = 134,22
Update = 21700, Epoch = 34, LR = 0,000407, AvgCost = 199,5463, Sent = 611, SentPerMin = 143,18, WordPerSec = 131,88
Update = 21800, Epoch = 34, LR = 0,000406, AvgCost = 199,0328, Sent = 906, SentPerMin = 144,29, WordPerSec = 133,11
Update = 21830, Epoch = 34, LR = 0,000406, AvgCost = 197,7449, Sent = 1000, SentPerMin = 145,32, WordPerSec = 133,37
Update = 21900, Epoch = 35, LR = 0,000405, AvgCost = 201,7502, Sent = 200, SentPerMin = 104,83, WordPerSec = 96,20
Update = 22000, Epoch = 35, LR = 0,000405, AvgCost = 199,2280, Sent = 501, SentPerMin = 104,03, WordPerSec = 95,30
Update = 22100, Epoch = 35, LR = 0,000404, AvgCost = 199,4737, Sent = 795, SentPerMin = 104,38, WordPerSec = 96,18
Update = 22168, Epoch = 35, LR = 0,000403, AvgCost = 197,8798, Sent = 1000, SentPerMin = 105,27, WordPerSec = 96,61
Update = 22200, Epoch = 36, LR = 0,000403, AvgCost = 191,7202, Sent = 99, SentPerMin = 147,50, WordPerSec = 128,53
Update = 22300, Epoch = 36, LR = 0,000402, AvgCost = 198,8328, Sent = 388, SentPerMin = 144,27, WordPerSec = 132,23
Update = 22400, Epoch = 36, LR = 0,000401, AvgCost = 197,0939, Sent = 683, SentPerMin = 145,03, WordPerSec = 132,89
Update = 22500, Epoch = 36, LR = 0,000400, AvgCost = 196,7493, Sent = 984, SentPerMin = 145,37, WordPerSec = 133,41
Update = 22506, Epoch = 36, LR = 0,000400, AvgCost = 196,6615, Sent = 1000, SentPerMin = 145,46, WordPerSec = 133,50
Update = 22600, Epoch = 37, LR = 0,000399, AvgCost = 204,4563, Sent = 269, SentPerMin = 103,87, WordPerSec = 97,08
Update = 22700, Epoch = 37, LR = 0,000398, AvgCost = 198,0322, Sent = 573, SentPerMin = 104,25, WordPerSec = 95,75
Update = 22800, Epoch = 37, LR = 0,000397, AvgCost = 197,5039, Sent = 865, SentPerMin = 104,80, WordPerSec = 96,52
Update = 22844, Epoch = 37, LR = 0,000397, AvgCost = 196,7204, Sent = 1000, SentPerMin = 105,31, WordPerSec = 96,65
Update = 22900, Epoch = 38, LR = 0,000396, AvgCost = 199,3261, Sent = 161, SentPerMin = 143,45, WordPerSec = 131,58
Update = 23000, Epoch = 38, LR = 0,000396, AvgCost = 200,1989, Sent = 456, SentPerMin = 142,92, WordPerSec = 132,05
Update = 23100, Epoch = 38, LR = 0,000395, AvgCost = 196,6797, Sent = 757, SentPerMin = 143,09, WordPerSec = 131,62
Update = 23182, Epoch = 38, LR = 0,000394, AvgCost = 195,6923, Sent = 1000, SentPerMin = 145,26, WordPerSec = 133,32
Update = 23200, Epoch = 39, LR = 0,000394, AvgCost = 184,5556, Sent = 59, SentPerMin = 112,87, WordPerSec = 94,31
Update = 23300, Epoch = 39, LR = 0,000393, AvgCost = 200,6996, Sent = 343, SentPerMin = 103,88, WordPerSec = 96,05
Update = 23400, Epoch = 39, LR = 0,000392, AvgCost = 197,0818, Sent = 640, SentPerMin = 104,61, WordPerSec = 96,13
Update = 23500, Epoch = 39, LR = 0,000391, AvgCost = 195,9591, Sent = 943, SentPerMin = 104,50, WordPerSec = 95,92
Update = 23520, Epoch = 39, LR = 0,000391, AvgCost = 195,9591, Sent = 1000, SentPerMin = 104,78, WordPerSec = 96,16
Starting inference...
Inference results:
在   (••
在   (••
在在  ( (在
在•••••  •••••  •••••••••••••••••••••••••••••••••••••••••••••••
在在    •


             */


            /*
             Update = 10100, Epoch = 0, LR = 0,000597, AvgCost = 266,8505, Sent = 285, SentPerMin = 623,91, WordPerSec = 585,86
Update = 10200, Epoch = 0, LR = 0,000594, AvgCost = 248,2995, Sent = 592, SentPerMin = 661,66, WordPerSec = 606,95
Update = 10300, Epoch = 0, LR = 0,000591, AvgCost = 246,0384, Sent = 882, SentPerMin = 669,65, WordPerSec = 618,64
Update = 10338, Epoch = 0, LR = 0,000590, AvgCost = 242,8491, Sent = 1000, SentPerMin = 676,34, WordPerSec = 620,72
Update = 10400, Epoch = 1, LR = 0,000588, AvgCost = 234,2564, Sent = 178, SentPerMin = 960,05, WordPerSec = 885,89
Update = 10500, Epoch = 1, LR = 0,000586, AvgCost = 234,4435, Sent = 475, SentPerMin = 968,53, WordPerSec = 892,14
Update = 10600, Epoch = 1, LR = 0,000583, AvgCost = 232,7469, Sent = 771, SentPerMin = 970,92, WordPerSec = 897,42
Update = 10676, Epoch = 1, LR = 0,000581, AvgCost = 229,4540, Sent = 1000, SentPerMin = 982,64, WordPerSec = 901,83
Update = 10700, Epoch = 2, LR = 0,000580, AvgCost = 221,5057, Sent = 76, SentPerMin = 981,69, WordPerSec = 865,86
Update = 10800, Epoch = 2, LR = 0,000577, AvgCost = 230,9157, Sent = 363, SentPerMin = 962,94, WordPerSec = 887,96
Update = 10900, Epoch = 2, LR = 0,000575, AvgCost = 227,7379, Sent = 656, SentPerMin = 970,26, WordPerSec = 893,35
Update = 11000, Epoch = 2, LR = 0,000572, AvgCost = 225,2310, Sent = 961, SentPerMin = 980,87, WordPerSec = 899,02
Update = 11014, Epoch = 2, LR = 0,000572, AvgCost = 225,6130, Sent = 1000, SentPerMin = 981,08, WordPerSec = 900,41
Update = 11100, Epoch = 3, LR = 0,000569, AvgCost = 228,8096, Sent = 248, SentPerMin = 966,61, WordPerSec = 892,95
Update = 11200, Epoch = 3, LR = 0,000567, AvgCost = 224,8617, Sent = 549, SentPerMin = 978,32, WordPerSec = 896,37
Update = 11300, Epoch = 3, LR = 0,000564, AvgCost = 224,8996, Sent = 841, SentPerMin = 978,21, WordPerSec = 900,79
Update = 11352, Epoch = 3, LR = 0,000563, AvgCost = 223,4520, Sent = 1000, SentPerMin = 984,70, WordPerSec = 903,73
Update = 11400, Epoch = 4, LR = 0,000562, AvgCost = 221,6687, Sent = 140, SentPerMin = 970,04, WordPerSec = 870,72
Update = 11500, Epoch = 4, LR = 0,000560, AvgCost = 228,4872, Sent = 431, SentPerMin = 960,02, WordPerSec = 889,90
Update = 11600, Epoch = 4, LR = 0,000557, AvgCost = 221,6719, Sent = 732, SentPerMin = 978,52, WordPerSec = 894,62
Update = 11690, Epoch = 4, LR = 0,000555, AvgCost = 221,7998, Sent = 1000, SentPerMin = 982,04, WordPerSec = 901,28
Update = 11700, Epoch = 5, LR = 0,000555, AvgCost = 183,0104, Sent = 33, SentPerMin = 1071,32, WordPerSec = 827,84
Update = 11800, Epoch = 5, LR = 0,000552, AvgCost = 228,2981, Sent = 317, SentPerMin = 960,08, WordPerSec = 894,06
Update = 11900, Epoch = 5, LR = 0,000550, AvgCost = 222,2892, Sent = 618, SentPerMin = 975,21, WordPerSec = 895,02
Update = 12000, Epoch = 5, LR = 0,000548, AvgCost = 221,7275, Sent = 914, SentPerMin = 976,36, WordPerSec = 899,72
Update = 12028, Epoch = 5, LR = 0,000547, AvgCost = 220,4737, Sent = 1000, SentPerMin = 981,37, WordPerSec = 900,67
Update = 12100, Epoch = 6, LR = 0,000545, AvgCost = 227,6427, Sent = 205, SentPerMin = 946,06, WordPerSec = 883,29
Update = 12200, Epoch = 6, LR = 0,000543, AvgCost = 221,3107, Sent = 506, SentPerMin = 967,05, WordPerSec = 887,00
Update = 12300, Epoch = 6, LR = 0,000541, AvgCost = 221,4247, Sent = 802, SentPerMin = 970,78, WordPerSec = 893,74
Update = 12366, Epoch = 6, LR = 0,000540, AvgCost = 219,4275, Sent = 1000, SentPerMin = 979,08, WordPerSec = 898,56
Update = 12400, Epoch = 7, LR = 0,000539, AvgCost = 209,0905, Sent = 106, SentPerMin = 1013,51, WordPerSec = 872,96
Update = 12500, Epoch = 7, LR = 0,000537, AvgCost = 222,2764, Sent = 393, SentPerMin = 966,10, WordPerSec = 890,38
Update = 12600, Epoch = 7, LR = 0,000535, AvgCost = 219,2858, Sent = 689, SentPerMin = 981,40, WordPerSec = 899,61
Update = 12700, Epoch = 7, LR = 0,000532, AvgCost = 218,8596, Sent = 989, SentPerMin = 982,76, WordPerSec = 902,19
Update = 12704, Epoch = 7, LR = 0,000532, AvgCost = 218,6837, Sent = 1000, SentPerMin = 982,60, WordPerSec = 901,80
Update = 12800, Epoch = 8, LR = 0,000530, AvgCost = 228,6535, Sent = 275, SentPerMin = 948,04, WordPerSec = 889,95
Update = 12900, Epoch = 8, LR = 0,000528, AvgCost = 219,1709, Sent = 581, SentPerMin = 971,98, WordPerSec = 891,65
Update = 13000, Epoch = 8, LR = 0,000526, AvgCost = 219,5178, Sent = 870, SentPerMin = 972,86, WordPerSec = 898,21
Update = 13042, Epoch = 8, LR = 0,000525, AvgCost = 217,8859, Sent = 1000, SentPerMin = 980,99, WordPerSec = 900,32
Update = 13100, Epoch = 9, LR = 0,000524, AvgCost = 220,2876, Sent = 166, SentPerMin = 948,79, WordPerSec = 874,30
Update = 13200, Epoch = 9, LR = 0,000522, AvgCost = 221,3300, Sent = 463, SentPerMin = 959,47, WordPerSec = 885,59
Update = 13300, Epoch = 9, LR = 0,000520, AvgCost = 219,3865, Sent = 761, SentPerMin = 966,29, WordPerSec = 889,81
Update = 13380, Epoch = 9, LR = 0,000519, AvgCost = 217,2374, Sent = 1000, SentPerMin = 976,86, WordPerSec = 896,53
Update = 13400, Epoch = 10, LR = 0,000518, AvgCost = 215,4456, Sent = 64, SentPerMin = 732,14, WordPerSec = 638,34
Update = 13500, Epoch = 10, LR = 0,000516, AvgCost = 222,1010, Sent = 352, SentPerMin = 682,49, WordPerSec = 627,62
Update = 13600, Epoch = 10, LR = 0,000514, AvgCost = 218,8336, Sent = 645, SentPerMin = 683,26, WordPerSec = 628,91
Update = 13700, Epoch = 10, LR = 0,000513, AvgCost = 217,6420, Sent = 948, SentPerMin = 689,84, WordPerSec = 634,17
Update = 13718, Epoch = 10, LR = 0,000512, AvgCost = 217,0975, Sent = 1000, SentPerMin = 690,76, WordPerSec = 633,96
Update = 13800, Epoch = 11, LR = 0,000511, AvgCost = 219,9149, Sent = 239, SentPerMin = 981,13, WordPerSec = 896,02
Update = 13900, Epoch = 11, LR = 0,000509, AvgCost = 219,1099, Sent = 536, SentPerMin = 975,56, WordPerSec = 896,72
Update = 14000, Epoch = 11, LR = 0,000507, AvgCost = 217,3715, Sent = 831, SentPerMin = 978,28, WordPerSec = 898,29
Update = 14056, Epoch = 11, LR = 0,000506, AvgCost = 216,4516, Sent = 1000, SentPerMin = 983,59, WordPerSec = 902,71
Update = 14100, Epoch = 12, LR = 0,000505, AvgCost = 215,3035, Sent = 131, SentPerMin = 982,80, WordPerSec = 874,64
Update = 14200, Epoch = 12, LR = 0,000504, AvgCost = 218,7263, Sent = 423, SentPerMin = 968,80, WordPerSec = 890,32
Update = 14300, Epoch = 12, LR = 0,000502, AvgCost = 215,6303, Sent = 720, SentPerMin = 982,13, WordPerSec = 898,80
Update = 14394, Epoch = 12, LR = 0,000500, AvgCost = 215,5851, Sent = 1000, SentPerMin = 984,62, WordPerSec = 903,65
Update = 14400, Epoch = 13, LR = 0,000500, AvgCost = 217,1646, Sent = 16, SentPerMin = 859,70, WordPerSec = 839,10
Update = 14500, Epoch = 13, LR = 0,000498, AvgCost = 223,6697, Sent = 304, SentPerMin = 956,43, WordPerSec = 893,30
Update = 14600, Epoch = 13, LR = 0,000497, AvgCost = 216,3165, Sent = 606, SentPerMin = 971,42, WordPerSec = 892,79
Update = 14700, Epoch = 13, LR = 0,000495, AvgCost = 215,7221, Sent = 902, SentPerMin = 977,23, WordPerSec = 900,02
Update = 14732, Epoch = 13, LR = 0,000494, AvgCost = 214,4903, Sent = 1000, SentPerMin = 982,53, WordPerSec = 901,73
Update = 14800, Epoch = 14, LR = 0,000493, AvgCost = 220,5512, Sent = 195, SentPerMin = 957,67, WordPerSec = 880,24
Update = 14900, Epoch = 14, LR = 0,000492, AvgCost = 215,6349, Sent = 495, SentPerMin = 974,55, WordPerSec = 890,65
Update = 15000, Epoch = 14, LR = 0,000490, AvgCost = 216,3123, Sent = 789, SentPerMin = 971,73, WordPerSec = 896,01
Update = 15070, Epoch = 14, LR = 0,000489, AvgCost = 213,8592, Sent = 1000, SentPerMin = 981,72, WordPerSec = 900,99
Update = 15100, Epoch = 15, LR = 0,000488, AvgCost = 211,7265, Sent = 92, SentPerMin = 977,29, WordPerSec = 867,34
Update = 15200, Epoch = 15, LR = 0,000487, AvgCost = 217,0660, Sent = 381, SentPerMin = 956,33, WordPerSec = 880,03
Update = 15300, Epoch = 15, LR = 0,000485, AvgCost = 214,1998, Sent = 676, SentPerMin = 970,06, WordPerSec = 890,58
Update = 15400, Epoch = 15, LR = 0,000483, AvgCost = 211,8502, Sent = 979, SentPerMin = 980,12, WordPerSec = 897,46
Update = 15408, Epoch = 15, LR = 0,000483, AvgCost = 212,9242, Sent = 1000, SentPerMin = 978,24, WordPerSec = 897,80
Update = 15500, Epoch = 16, LR = 0,000482, AvgCost = 220,4401, Sent = 265, SentPerMin = 959,46, WordPerSec = 890,42
Update = 15600, Epoch = 16, LR = 0,000480, AvgCost = 214,3056, Sent = 566, SentPerMin = 974,19, WordPerSec = 896,62
Update = 15700, Epoch = 16, LR = 0,000479, AvgCost = 213,4631, Sent = 859, SentPerMin = 974,87, WordPerSec = 898,34
Update = 15746, Epoch = 16, LR = 0,000478, AvgCost = 212,1350, Sent = 1000, SentPerMin = 983,16, WordPerSec = 902,31
Update = 15800, Epoch = 17, LR = 0,000477, AvgCost = 214,1721, Sent = 157, SentPerMin = 965,77, WordPerSec = 878,53
Update = 15900, Epoch = 17, LR = 0,000476, AvgCost = 215,8510, Sent = 450, SentPerMin = 965,37, WordPerSec = 890,18
Update = 16000, Epoch = 17, LR = 0,000474, AvgCost = 211,1838, Sent = 752, SentPerMin = 973,10, WordPerSec = 892,16
Update = 16084, Epoch = 17, LR = 0,000473, AvgCost = 211,0851, Sent = 1000, SentPerMin = 980,07, WordPerSec = 899,47
Update = 16100, Epoch = 18, LR = 0,000473, AvgCost = 204,4361, Sent = 51, SentPerMin = 1009,88, WordPerSec = 870,61
Update = 16200, Epoch = 18, LR = 0,000471, AvgCost = 216,3886, Sent = 336, SentPerMin = 956,73, WordPerSec = 886,40
Update = 16300, Epoch = 18, LR = 0,000470, AvgCost = 210,7436, Sent = 636, SentPerMin = 971,57, WordPerSec = 890,17
Update = 16400, Epoch = 18, LR = 0,000469, AvgCost = 210,5040, Sent = 935, SentPerMin = 975,08, WordPerSec = 896,35
Update = 16422, Epoch = 18, LR = 0,000468, AvgCost = 210,0748, Sent = 1000, SentPerMin = 978,95, WordPerSec = 898,45
Update = 16500, Epoch = 19, LR = 0,000467, AvgCost = 213,2034, Sent = 225, SentPerMin = 963,14, WordPerSec = 885,38
Update = 16600, Epoch = 19, LR = 0,000466, AvgCost = 212,0532, Sent = 524, SentPerMin = 966,83, WordPerSec = 889,62
Update = 16700, Epoch = 19, LR = 0,000464, AvgCost = 210,5182, Sent = 819, SentPerMin = 973,39, WordPerSec = 895,32
Update = 16760, Epoch = 19, LR = 0,000463, AvgCost = 209,2816, Sent = 1000, SentPerMin = 980,75, WordPerSec = 900,10
Update = 16800, Epoch = 20, LR = 0,000463, AvgCost = 206,9759, Sent = 122, SentPerMin = 716,13, WordPerSec = 625,05
Update = 16900, Epoch = 20, LR = 0,000462, AvgCost = 213,0421, Sent = 411, SentPerMin = 681,46, WordPerSec = 627,63
Update = 17000, Epoch = 20, LR = 0,000460, AvgCost = 209,4662, Sent = 709, SentPerMin = 691,68, WordPerSec = 631,81
Update = 17098, Epoch = 20, LR = 0,000459, AvgCost = 209,8431, Sent = 1000, SentPerMin = 689,30, WordPerSec = 632,61
Update = 17100, Epoch = 21, LR = 0,000459, AvgCost = 186,2997, Sent = 6, SentPerMin = 931,26, WordPerSec = 845,89
Update = 17200, Epoch = 21, LR = 0,000457, AvgCost = 217,7037, Sent = 292, SentPerMin = 944,50, WordPerSec = 883,69
Update = 17300, Epoch = 21, LR = 0,000456, AvgCost = 210,6440, Sent = 595, SentPerMin = 965,74, WordPerSec = 888,86
Update = 17400, Epoch = 21, LR = 0,000455, AvgCost = 210,0192, Sent = 890, SentPerMin = 972,80, WordPerSec = 896,74
Update = 17436, Epoch = 21, LR = 0,000454, AvgCost = 208,6503, Sent = 1000, SentPerMin = 978,92, WordPerSec = 898,42
Update = 17500, Epoch = 22, LR = 0,000454, AvgCost = 214,4864, Sent = 184, SentPerMin = 959,98, WordPerSec = 881,98
Update = 17600, Epoch = 22, LR = 0,000452, AvgCost = 210,4581, Sent = 482, SentPerMin = 968,09, WordPerSec = 890,03
Update = 17700, Epoch = 22, LR = 0,000451, AvgCost = 210,8658, Sent = 775, SentPerMin = 966,56, WordPerSec = 895,53
Update = 17774, Epoch = 22, LR = 0,000450, AvgCost = 207,6884, Sent = 1000, SentPerMin = 981,25, WordPerSec = 900,56
Update = 17800, Epoch = 23, LR = 0,000450, AvgCost = 202,8316, Sent = 81, SentPerMin = 987,99, WordPerSec = 876,99
Update = 17900, Epoch = 23, LR = 0,000448, AvgCost = 211,9231, Sent = 368, SentPerMin = 958,17, WordPerSec = 886,74
Update = 18000, Epoch = 23, LR = 0,000447, AvgCost = 207,9826, Sent = 663, SentPerMin = 972,82, WordPerSec = 893,81
Update = 18100, Epoch = 23, LR = 0,000446, AvgCost = 205,8588, Sent = 969, SentPerMin = 983,24, WordPerSec = 899,10
Update = 18112, Epoch = 23, LR = 0,000446, AvgCost = 206,7446, Sent = 1000, SentPerMin = 980,54, WordPerSec = 899,91
Update = 18200, Epoch = 24, LR = 0,000445, AvgCost = 216,0264, Sent = 251, SentPerMin = 950,75, WordPerSec = 888,19
Update = 18300, Epoch = 24, LR = 0,000444, AvgCost = 206,7060, Sent = 556, SentPerMin = 975,25, WordPerSec = 892,66
Update = 18400, Epoch = 24, LR = 0,000442, AvgCost = 207,3716, Sent = 847, SentPerMin = 972,17, WordPerSec = 896,06
Update = 18450, Epoch = 24, LR = 0,000442, AvgCost = 206,0684, Sent = 1000, SentPerMin = 980,68, WordPerSec = 900,04
Update = 18500, Epoch = 25, LR = 0,000441, AvgCost = 208,7524, Sent = 145, SentPerMin = 957,72, WordPerSec = 871,09
Update = 18600, Epoch = 25, LR = 0,000440, AvgCost = 210,8745, Sent = 437, SentPerMin = 960,33, WordPerSec = 889,94
Update = 18700, Epoch = 25, LR = 0,000439, AvgCost = 205,4465, Sent = 739, SentPerMin = 975,73, WordPerSec = 894,44
Update = 18788, Epoch = 25, LR = 0,000438, AvgCost = 205,3933, Sent = 1000, SentPerMin = 982,00, WordPerSec = 901,25
Update = 18800, Epoch = 26, LR = 0,000438, AvgCost = 167,8755, Sent = 41, SentPerMin = 1082,26, WordPerSec = 833,25
Update = 18900, Epoch = 26, LR = 0,000436, AvgCost = 211,4394, Sent = 324, SentPerMin = 953,96, WordPerSec = 886,09
Update = 19000, Epoch = 26, LR = 0,000435, AvgCost = 205,7665, Sent = 624, SentPerMin = 971,93, WordPerSec = 891,02
Update = 19100, Epoch = 26, LR = 0,000434, AvgCost = 205,4245, Sent = 921, SentPerMin = 976,11, WordPerSec = 898,07
Update = 19126, Epoch = 26, LR = 0,000434, AvgCost = 204,6983, Sent = 1000, SentPerMin = 980,29, WordPerSec = 899,68
Update = 19200, Epoch = 27, LR = 0,000433, AvgCost = 209,9686, Sent = 213, SentPerMin = 958,02, WordPerSec = 883,81
Update = 19300, Epoch = 27, LR = 0,000432, AvgCost = 207,4480, Sent = 511, SentPerMin = 964,06, WordPerSec = 888,97
Update = 19400, Epoch = 27, LR = 0,000431, AvgCost = 205,8413, Sent = 808, SentPerMin = 973,27, WordPerSec = 896,50
Update = 19464, Epoch = 27, LR = 0,000430, AvgCost = 204,0428, Sent = 1000, SentPerMin = 979,35, WordPerSec = 898,82
Update = 19500, Epoch = 28, LR = 0,000430, AvgCost = 194,1948, Sent = 113, SentPerMin = 1024,78, WordPerSec = 877,86
Update = 19600, Epoch = 28, LR = 0,000429, AvgCost = 206,4099, Sent = 399, SentPerMin = 963,53, WordPerSec = 886,90
Update = 19700, Epoch = 28, LR = 0,000427, AvgCost = 203,8700, Sent = 696, SentPerMin = 976,31, WordPerSec = 894,51
Update = 19800, Epoch = 28, LR = 0,000426, AvgCost = 203,4831, Sent = 995, SentPerMin = 978,35, WordPerSec = 897,28
Update = 19802, Epoch = 28, LR = 0,000426, AvgCost = 203,4304, Sent = 1000, SentPerMin = 977,78, WordPerSec = 897,37
Update = 19900, Epoch = 29, LR = 0,000425, AvgCost = 213,0415, Sent = 280, SentPerMin = 939,74, WordPerSec = 881,79
Update = 20000, Epoch = 29, LR = 0,000424, AvgCost = 203,5432, Sent = 588, SentPerMin = 968,83, WordPerSec = 886,23
Update = 20100, Epoch = 29, LR = 0,000423, AvgCost = 204,2043, Sent = 877, SentPerMin = 716,81, WordPerSec = 661,08
Update = 20140, Epoch = 29, LR = 0,000423, AvgCost = 203,0022, Sent = 1000, SentPerMin = 745,01, WordPerSec = 683,74
Update = 20200, Epoch = 30, LR = 0,000422, AvgCost = 209,2808, Sent = 173, SentPerMin = 951,05, WordPerSec = 875,00
Update = 20300, Epoch = 30, LR = 0,000421, AvgCost = 208,9407, Sent = 469, SentPerMin = 789,79, WordPerSec = 728,52
Update = 20400, Epoch = 30, LR = 0,000420, AvgCost = 207,2043, Sent = 766, SentPerMin = 745,53, WordPerSec = 687,93
Update = 20478, Epoch = 30, LR = 0,000419, AvgCost = 204,7184, Sent = 1000, SentPerMin = 738,54, WordPerSec = 677,81
Update = 20500, Epoch = 31, LR = 0,000419, AvgCost = 197,9660, Sent = 71, SentPerMin = 1012,46, WordPerSec = 876,27
Update = 20600, Epoch = 31, LR = 0,000418, AvgCost = 207,7447, Sent = 358, SentPerMin = 960,59, WordPerSec = 882,37
Update = 20700, Epoch = 31, LR = 0,000417, AvgCost = 204,7223, Sent = 652, SentPerMin = 968,14, WordPerSec = 889,02
Update = 20800, Epoch = 31, LR = 0,000416, AvgCost = 203,2739, Sent = 955, SentPerMin = 976,87, WordPerSec = 896,21
Update = 20816, Epoch = 31, LR = 0,000416, AvgCost = 203,3421, Sent = 1000, SentPerMin = 976,86, WordPerSec = 896,53
Update = 20900, Epoch = 32, LR = 0,000415, AvgCost = 206,6005, Sent = 244, SentPerMin = 970,79, WordPerSec = 886,91
Update = 21000, Epoch = 32, LR = 0,000414, AvgCost = 204,6993, Sent = 542, SentPerMin = 966,11, WordPerSec = 887,32
Update = 21100, Epoch = 32, LR = 0,000413, AvgCost = 203,7162, Sent = 835, SentPerMin = 966,58, WordPerSec = 889,28
Update = 21154, Epoch = 32, LR = 0,000413, AvgCost = 202,5731, Sent = 1000, SentPerMin = 973,82, WordPerSec = 893,74
Update = 21200, Epoch = 33, LR = 0,000412, AvgCost = 202,8033, Sent = 136, SentPerMin = 984,18, WordPerSec = 875,75
Update = 21300, Epoch = 33, LR = 0,000411, AvgCost = 205,4683, Sent = 428, SentPerMin = 964,18, WordPerSec = 886,84
Update = 21400, Epoch = 33, LR = 0,000410, AvgCost = 202,4805, Sent = 725, SentPerMin = 974,51, WordPerSec = 893,30
Update = 21492, Epoch = 33, LR = 0,000409, AvgCost = 202,1071, Sent = 1000, SentPerMin = 977,31, WordPerSec = 896,94
Update = 21500, Epoch = 34, LR = 0,000409, AvgCost = 187,9564, Sent = 23, SentPerMin = 935,35, WordPerSec = 815,38
Update = 21600, Epoch = 34, LR = 0,000408, AvgCost = 209,1693, Sent = 311, SentPerMin = 948,21, WordPerSec = 881,60
Update = 21700, Epoch = 34, LR = 0,000407, AvgCost = 203,6285, Sent = 611, SentPerMin = 963,80, WordPerSec = 887,69
Update = 21800, Epoch = 34, LR = 0,000406, AvgCost = 202,8372, Sent = 906, SentPerMin = 969,31, WordPerSec = 894,19
Update = 21830, Epoch = 34, LR = 0,000406, AvgCost = 201,5006, Sent = 1000, SentPerMin = 976,46, WordPerSec = 896,16
Update = 21900, Epoch = 35, LR = 0,000405, AvgCost = 207,6365, Sent = 200, SentPerMin = 954,74, WordPerSec = 876,13
Update = 22000, Epoch = 35, LR = 0,000405, AvgCost = 202,9102, Sent = 501, SentPerMin = 967,00, WordPerSec = 885,90
Update = 22100, Epoch = 35, LR = 0,000404, AvgCost = 203,0466, Sent = 795, SentPerMin = 967,16, WordPerSec = 891,14
Update = 22168, Epoch = 35, LR = 0,000403, AvgCost = 201,0165, Sent = 1000, SentPerMin = 976,29, WordPerSec = 896,01
Update = 22200, Epoch = 36, LR = 0,000403, AvgCost = 198,0218, Sent = 99, SentPerMin = 998,72, WordPerSec = 870,26
Update = 22300, Epoch = 36, LR = 0,000402, AvgCost = 203,3810, Sent = 388, SentPerMin = 965,55, WordPerSec = 884,97
Update = 22400, Epoch = 36, LR = 0,000401, AvgCost = 201,2072, Sent = 683, SentPerMin = 972,58, WordPerSec = 891,18
Update = 22500, Epoch = 36, LR = 0,000400, AvgCost = 200,6486, Sent = 984, SentPerMin = 977,13, WordPerSec = 896,72
Update = 22506, Epoch = 36, LR = 0,000400, AvgCost = 200,5114, Sent = 1000, SentPerMin = 976,83, WordPerSec = 896,50
Update = 22600, Epoch = 37, LR = 0,000399, AvgCost = 209,4061, Sent = 269, SentPerMin = 951,34, WordPerSec = 889,21
Update = 22700, Epoch = 37, LR = 0,000398, AvgCost = 201,5239, Sent = 573, SentPerMin = 969,82, WordPerSec = 890,69
Update = 22800, Epoch = 37, LR = 0,000397, AvgCost = 200,7499, Sent = 865, SentPerMin = 969,62, WordPerSec = 893,06
Update = 22844, Epoch = 37, LR = 0,000397, AvgCost = 199,7506, Sent = 1000, SentPerMin = 976,76, WordPerSec = 896,43
Update = 22900, Epoch = 38, LR = 0,000396, AvgCost = 204,9404, Sent = 161, SentPerMin = 950,48, WordPerSec = 871,86
Update = 23000, Epoch = 38, LR = 0,000396, AvgCost = 203,8053, Sent = 456, SentPerMin = 958,42, WordPerSec = 885,56
Update = 23100, Epoch = 38, LR = 0,000395, AvgCost = 200,2641, Sent = 757, SentPerMin = 967,36, WordPerSec = 889,79
Update = 23182, Epoch = 38, LR = 0,000394, AvgCost = 199,1045, Sent = 1000, SentPerMin = 973,69, WordPerSec = 893,62
Update = 23200, Epoch = 39, LR = 0,000394, AvgCost = 191,7823, Sent = 59, SentPerMin = 752,73, WordPerSec = 628,98
Update = 23300, Epoch = 39, LR = 0,000393, AvgCost = 205,1390, Sent = 343, SentPerMin = 758,74, WordPerSec = 701,52
Update = 23400, Epoch = 39, LR = 0,000392, AvgCost = 200,1131, Sent = 640, SentPerMin = 840,09, WordPerSec = 771,99
Update = 23500, Epoch = 39, LR = 0,000391, AvgCost = 198,8068, Sent = 943, SentPerMin = 878,28, WordPerSec = 806,16
Update = 23520, Epoch = 39, LR = 0,000391, AvgCost = 198,7633, Sent = 1000, SentPerMin = 882,91, WordPerSec = 810,31
Update = 23600, Epoch = 40, LR = 0,000391, AvgCost = 203,2651, Sent = 231, SentPerMin = 927,72, WordPerSec = 853,69
Update = 23700, Epoch = 40, LR = 0,000390, AvgCost = 201,6762, Sent = 529, SentPerMin = 767,31, WordPerSec = 706,22
Update = 23800, Epoch = 40, LR = 0,000389, AvgCost = 200,4071, Sent = 824, SentPerMin = 734,69, WordPerSec = 676,06
Update = 23858, Epoch = 40, LR = 0,000388, AvgCost = 199,2462, Sent = 1000, SentPerMin = 729,75, WordPerSec = 669,74
Update = 23900, Epoch = 41, LR = 0,000388, AvgCost = 198,9446, Sent = 126, SentPerMin = 992,21, WordPerSec = 875,93
Update = 24000, Epoch = 41, LR = 0,000387, AvgCost = 200,6622, Sent = 418, SentPerMin = 962,68, WordPerSec = 883,38
Update = 24100, Epoch = 41, LR = 0,000386, AvgCost = 197,7184, Sent = 715, SentPerMin = 975,10, WordPerSec = 891,73
Update = 24196, Epoch = 41, LR = 0,000386, AvgCost = 198,1589, Sent = 1000, SentPerMin = 976,99, WordPerSec = 896,65
Update = 24200, Epoch = 42, LR = 0,000386, AvgCost = 179,4330, Sent = 12, SentPerMin = 983,01, WordPerSec = 884,71
Update = 24300, Epoch = 42, LR = 0,000385, AvgCost = 206,0891, Sent = 299, SentPerMin = 947,58, WordPerSec = 883,88
Update = 24400, Epoch = 42, LR = 0,000384, AvgCost = 199,0552, Sent = 601, SentPerMin = 966,94, WordPerSec = 888,26
Update = 24500, Epoch = 42, LR = 0,000383, AvgCost = 198,2897, Sent = 897, SentPerMin = 970,16, WordPerSec = 892,38
Update = 24534, Epoch = 42, LR = 0,000383, AvgCost = 197,5733, Sent = 1000, SentPerMin = 974,11, WordPerSec = 894,00
Update = 24600, Epoch = 43, LR = 0,000383, AvgCost = 204,5866, Sent = 189, SentPerMin = 947,69, WordPerSec = 872,98
Update = 24700, Epoch = 43, LR = 0,000382, AvgCost = 198,5279, Sent = 490, SentPerMin = 969,63, WordPerSec = 886,42
Update = 24800, Epoch = 43, LR = 0,000381, AvgCost = 199,0001, Sent = 784, SentPerMin = 966,72, WordPerSec = 890,95
Update = 24872, Epoch = 43, LR = 0,000380, AvgCost = 196,8215, Sent = 1000, SentPerMin = 975,93, WordPerSec = 895,68
Update = 24900, Epoch = 44, LR = 0,000380, AvgCost = 193,3799, Sent = 87, SentPerMin = 999,92, WordPerSec = 878,67
Update = 25000, Epoch = 44, LR = 0,000379, AvgCost = 200,3911, Sent = 375, SentPerMin = 962,41, WordPerSec = 887,04
Update = 25100, Epoch = 44, LR = 0,000379, AvgCost = 198,0016, Sent = 668, SentPerMin = 967,37, WordPerSec = 891,42
Update = 25200, Epoch = 44, LR = 0,000378, AvgCost = 195,4167, Sent = 974, SentPerMin = 980,80, WordPerSec = 897,50
Update = 25210, Epoch = 44, LR = 0,000378, AvgCost = 196,4189, Sent = 1000, SentPerMin = 978,07, WordPerSec = 897,64
Update = 25300, Epoch = 45, LR = 0,000377, AvgCost = 204,6685, Sent = 258, SentPerMin = 943,83, WordPerSec = 879,75
Update = 25400, Epoch = 45, LR = 0,000376, AvgCost = 196,8230, Sent = 562, SentPerMin = 969,64, WordPerSec = 889,09
Update = 25500, Epoch = 45, LR = 0,000376, AvgCost = 197,1532, Sent = 854, SentPerMin = 969,43, WordPerSec = 892,73
Update = 25548, Epoch = 45, LR = 0,000375, AvgCost = 196,0603, Sent = 1000, SentPerMin = 975,98, WordPerSec = 895,72
Update = 25600, Epoch = 46, LR = 0,000375, AvgCost = 196,8297, Sent = 152, SentPerMin = 968,58, WordPerSec = 871,40
Update = 25700, Epoch = 46, LR = 0,000374, AvgCost = 200,1411, Sent = 443, SentPerMin = 959,33, WordPerSec = 886,97
Update = 25800, Epoch = 46, LR = 0,000374, AvgCost = 195,9604, Sent = 744, SentPerMin = 968,65, WordPerSec = 889,64
Update = 25886, Epoch = 46, LR = 0,000373, AvgCost = 195,6813, Sent = 1000, SentPerMin = 975,82, WordPerSec = 895,58
Update = 25900, Epoch = 47, LR = 0,000373, AvgCost = 186,1929, Sent = 45, SentPerMin = 1022,01, WordPerSec = 836,91
Update = 26000, Epoch = 47, LR = 0,000372, AvgCost = 201,3450, Sent = 330, SentPerMin = 954,01, WordPerSec = 884,34
Update = 26100, Epoch = 47, LR = 0,000371, AvgCost = 196,1228, Sent = 630, SentPerMin = 968,61, WordPerSec = 888,27
Update = 26200, Epoch = 47, LR = 0,000371, AvgCost = 195,4968, Sent = 929, SentPerMin = 973,39, WordPerSec = 894,30
Update = 26224, Epoch = 47, LR = 0,000371, AvgCost = 195,1259, Sent = 1000, SentPerMin = 976,78, WordPerSec = 896,45
Update = 26300, Epoch = 48, LR = 0,000370, AvgCost = 199,8414, Sent = 219, SentPerMin = 959,58, WordPerSec = 884,22
Update = 26400, Epoch = 48, LR = 0,000369, AvgCost = 196,9306, Sent = 519, SentPerMin = 961,07, WordPerSec = 881,91
Update = 26500, Epoch = 48, LR = 0,000369, AvgCost = 195,9671, Sent = 814, SentPerMin = 963,55, WordPerSec = 886,27
Update = 26562, Epoch = 48, LR = 0,000368, AvgCost = 194,6479, Sent = 1000, SentPerMin = 970,73, WordPerSec = 890,90
Update = 26600, Epoch = 49, LR = 0,000368, AvgCost = 184,4629, Sent = 119, SentPerMin = 739,06, WordPerSec = 628,72
Update = 26700, Epoch = 49, LR = 0,000367, AvgCost = 198,0902, Sent = 406, SentPerMin = 787,44, WordPerSec = 723,73
Update = 26800, Epoch = 49, LR = 0,000367, AvgCost = 194,9237, Sent = 702, SentPerMin = 855,63, WordPerSec = 782,50
Update = 26900, Epoch = 49, LR = 0,000366, AvgCost = 194,7668, Sent = 1000, SentPerMin = 884,76, WordPerSec = 812,01
Update = 26900, Epoch = 49, LR = 0,000366, AvgCost = 194,7668, Sent = 1000, SentPerMin = 884,76, WordPerSec = 812,00
Starting inference...
Inference results:
"""""""""""
"的•    的的和    和和和和
了•• ••"" "" 的    。  了•••••和和和和和和和和和。
了在在在的的的的了。"
"""    的 的 的     和。

             */


            /*
             Update = 10100, Epoch = 0, LR = 0,000597, AvgCost = 266,4576, Sent = 285, SentPerMin = 592,96, WordPerSec = 556,80
Update = 10200, Epoch = 0, LR = 0,000594, AvgCost = 248,4176, Sent = 592, SentPerMin = 630,65, WordPerSec = 578,51
Update = 10300, Epoch = 0, LR = 0,000591, AvgCost = 246,0991, Sent = 882, SentPerMin = 640,96, WordPerSec = 592,14
Update = 10338, Epoch = 0, LR = 0,000590, AvgCost = 242,8912, Sent = 1000, SentPerMin = 647,14, WordPerSec = 593,92
Update = 10400, Epoch = 1, LR = 0,000588, AvgCost = 235,2646, Sent = 178, SentPerMin = 920,12, WordPerSec = 849,04
Update = 10500, Epoch = 1, LR = 0,000586, AvgCost = 234,9106, Sent = 475, SentPerMin = 912,61, WordPerSec = 840,63
Update = 10600, Epoch = 1, LR = 0,000583, AvgCost = 233,0848, Sent = 771, SentPerMin = 920,36, WordPerSec = 850,69
Update = 10676, Epoch = 1, LR = 0,000581, AvgCost = 229,6916, Sent = 1000, SentPerMin = 935,34, WordPerSec = 858,42
Update = 10700, Epoch = 2, LR = 0,000580, AvgCost = 221,2294, Sent = 76, SentPerMin = 971,36, WordPerSec = 856,76
Update = 10800, Epoch = 2, LR = 0,000577, AvgCost = 230,8737, Sent = 363, SentPerMin = 931,79, WordPerSec = 859,24
Update = 10900, Epoch = 2, LR = 0,000575, AvgCost = 227,7396, Sent = 656, SentPerMin = 939,91, WordPerSec = 865,41
Update = 11000, Epoch = 2, LR = 0,000572, AvgCost = 225,1436, Sent = 961, SentPerMin = 955,94, WordPerSec = 876,18
Update = 11014, Epoch = 2, LR = 0,000572, AvgCost = 225,5184, Sent = 1000, SentPerMin = 956,09, WordPerSec = 877,47
Update = 11100, Epoch = 3, LR = 0,000569, AvgCost = 228,1548, Sent = 248, SentPerMin = 951,32, WordPerSec = 878,82
Update = 11200, Epoch = 3, LR = 0,000567, AvgCost = 224,4868, Sent = 549, SentPerMin = 963,47, WordPerSec = 882,77
Update = 11300, Epoch = 3, LR = 0,000564, AvgCost = 224,6564, Sent = 841, SentPerMin = 961,98, WordPerSec = 885,84
Update = 11352, Epoch = 3, LR = 0,000563, AvgCost = 223,1535, Sent = 1000, SentPerMin = 967,75, WordPerSec = 888,17
Update = 11400, Epoch = 4, LR = 0,000562, AvgCost = 220,5455, Sent = 140, SentPerMin = 961,50, WordPerSec = 863,06
Update = 11500, Epoch = 4, LR = 0,000560, AvgCost = 227,9330, Sent = 431, SentPerMin = 948,02, WordPerSec = 878,77
Update = 11600, Epoch = 4, LR = 0,000557, AvgCost = 221,1781, Sent = 732, SentPerMin = 964,84, WordPerSec = 882,11
Update = 11690, Epoch = 4, LR = 0,000555, AvgCost = 221,5330, Sent = 1000, SentPerMin = 964,41, WordPerSec = 885,10
Update = 11700, Epoch = 5, LR = 0,000555, AvgCost = 182,9080, Sent = 33, SentPerMin = 1029,45, WordPerSec = 795,48
Update = 11800, Epoch = 5, LR = 0,000552, AvgCost = 227,9502, Sent = 317, SentPerMin = 937,46, WordPerSec = 872,99
Update = 11900, Epoch = 5, LR = 0,000550, AvgCost = 221,8461, Sent = 618, SentPerMin = 954,93, WordPerSec = 876,41
Update = 12000, Epoch = 5, LR = 0,000548, AvgCost = 221,3668, Sent = 914, SentPerMin = 959,74, WordPerSec = 884,40
Update = 12028, Epoch = 5, LR = 0,000547, AvgCost = 220,2207, Sent = 1000, SentPerMin = 964,86, WordPerSec = 885,52
Update = 12100, Epoch = 6, LR = 0,000545, AvgCost = 228,0890, Sent = 205, SentPerMin = 922,86, WordPerSec = 861,64
Update = 12200, Epoch = 6, LR = 0,000543, AvgCost = 221,2716, Sent = 506, SentPerMin = 949,81, WordPerSec = 871,19
Update = 12300, Epoch = 6, LR = 0,000541, AvgCost = 221,1064, Sent = 802, SentPerMin = 954,56, WordPerSec = 878,80
Update = 12366, Epoch = 6, LR = 0,000540, AvgCost = 219,2360, Sent = 1000, SentPerMin = 962,96, WordPerSec = 883,77
Update = 12400, Epoch = 7, LR = 0,000539, AvgCost = 208,6800, Sent = 106, SentPerMin = 990,25, WordPerSec = 852,92
Update = 12500, Epoch = 7, LR = 0,000537, AvgCost = 222,0478, Sent = 393, SentPerMin = 943,43, WordPerSec = 869,49
Update = 12600, Epoch = 7, LR = 0,000535, AvgCost = 218,6829, Sent = 689, SentPerMin = 959,21, WordPerSec = 879,27
Update = 12700, Epoch = 7, LR = 0,000532, AvgCost = 218,4411, Sent = 989, SentPerMin = 962,73, WordPerSec = 883,80
Update = 12704, Epoch = 7, LR = 0,000532, AvgCost = 218,3208, Sent = 1000, SentPerMin = 962,89, WordPerSec = 883,70
Update = 12800, Epoch = 8, LR = 0,000530, AvgCost = 228,3786, Sent = 275, SentPerMin = 931,99, WordPerSec = 874,89
Update = 12900, Epoch = 8, LR = 0,000528, AvgCost = 218,4055, Sent = 581, SentPerMin = 955,31, WordPerSec = 876,36
Update = 13000, Epoch = 8, LR = 0,000526, AvgCost = 218,8344, Sent = 870, SentPerMin = 956,38, WordPerSec = 883,00
Update = 13042, Epoch = 8, LR = 0,000525, AvgCost = 217,4048, Sent = 1000, SentPerMin = 965,05, WordPerSec = 885,69
Update = 13100, Epoch = 9, LR = 0,000524, AvgCost = 220,0913, Sent = 166, SentPerMin = 931,49, WordPerSec = 858,36
Update = 13200, Epoch = 9, LR = 0,000522, AvgCost = 220,5037, Sent = 463, SentPerMin = 947,37, WordPerSec = 874,42
Update = 13300, Epoch = 9, LR = 0,000520, AvgCost = 218,6421, Sent = 761, SentPerMin = 951,87, WordPerSec = 876,52
Update = 13380, Epoch = 9, LR = 0,000519, AvgCost = 216,7642, Sent = 1000, SentPerMin = 963,82, WordPerSec = 884,56
Update = 13400, Epoch = 10, LR = 0,000518, AvgCost = 213,8459, Sent = 64, SentPerMin = 703,43, WordPerSec = 613,30
Update = 13500, Epoch = 10, LR = 0,000516, AvgCost = 220,8856, Sent = 352, SentPerMin = 671,59, WordPerSec = 617,60
Update = 13600, Epoch = 10, LR = 0,000514, AvgCost = 217,9941, Sent = 645, SentPerMin = 674,12, WordPerSec = 620,51
Update = 13700, Epoch = 10, LR = 0,000513, AvgCost = 217,0656, Sent = 948, SentPerMin = 679,63, WordPerSec = 624,79
Update = 13718, Epoch = 10, LR = 0,000512, AvgCost = 216,6814, Sent = 1000, SentPerMin = 680,78, WordPerSec = 624,80
Update = 13800, Epoch = 11, LR = 0,000511, AvgCost = 218,5341, Sent = 239, SentPerMin = 962,51, WordPerSec = 879,02
Update = 13900, Epoch = 11, LR = 0,000509, AvgCost = 218,0138, Sent = 536, SentPerMin = 956,51, WordPerSec = 879,21
Update = 14000, Epoch = 11, LR = 0,000507, AvgCost = 216,6601, Sent = 831, SentPerMin = 961,01, WordPerSec = 882,43
Update = 14056, Epoch = 11, LR = 0,000506, AvgCost = 215,9509, Sent = 1000, SentPerMin = 967,03, WordPerSec = 887,51
Update = 14100, Epoch = 12, LR = 0,000505, AvgCost = 214,3336, Sent = 131, SentPerMin = 965,60, WordPerSec = 859,33
Update = 14200, Epoch = 12, LR = 0,000504, AvgCost = 217,2580, Sent = 423, SentPerMin = 950,85, WordPerSec = 873,82
Update = 14300, Epoch = 12, LR = 0,000502, AvgCost = 214,8279, Sent = 720, SentPerMin = 962,22, WordPerSec = 880,58
Update = 14394, Epoch = 12, LR = 0,000500, AvgCost = 215,0885, Sent = 1000, SentPerMin = 964,10, WordPerSec = 884,82
Update = 14400, Epoch = 13, LR = 0,000500, AvgCost = 216,6602, Sent = 16, SentPerMin = 878,04, WordPerSec = 857,00
Update = 14500, Epoch = 13, LR = 0,000498, AvgCost = 222,0652, Sent = 304, SentPerMin = 942,19, WordPerSec = 879,99
Update = 14600, Epoch = 13, LR = 0,000497, AvgCost = 215,2423, Sent = 606, SentPerMin = 954,33, WordPerSec = 877,08
Update = 14700, Epoch = 13, LR = 0,000495, AvgCost = 215,0692, Sent = 902, SentPerMin = 957,32, WordPerSec = 881,68
Update = 14732, Epoch = 13, LR = 0,000494, AvgCost = 213,8749, Sent = 1000, SentPerMin = 962,77, WordPerSec = 883,60
Update = 14800, Epoch = 14, LR = 0,000493, AvgCost = 218,6556, Sent = 195, SentPerMin = 949,27, WordPerSec = 872,52
Update = 14900, Epoch = 14, LR = 0,000492, AvgCost = 214,1948, Sent = 495, SentPerMin = 964,04, WordPerSec = 881,04
Update = 15000, Epoch = 14, LR = 0,000490, AvgCost = 215,4015, Sent = 789, SentPerMin = 957,73, WordPerSec = 883,10
Update = 15070, Epoch = 14, LR = 0,000489, AvgCost = 213,0046, Sent = 1000, SentPerMin = 967,63, WordPerSec = 888,06
Update = 15100, Epoch = 15, LR = 0,000488, AvgCost = 211,5874, Sent = 92, SentPerMin = 974,27, WordPerSec = 864,66
Update = 15200, Epoch = 15, LR = 0,000487, AvgCost = 215,8170, Sent = 381, SentPerMin = 950,18, WordPerSec = 874,37
Update = 15300, Epoch = 15, LR = 0,000485, AvgCost = 213,2694, Sent = 676, SentPerMin = 956,62, WordPerSec = 878,25
Update = 15400, Epoch = 15, LR = 0,000483, AvgCost = 211,1219, Sent = 979, SentPerMin = 966,01, WordPerSec = 884,53
Update = 15408, Epoch = 15, LR = 0,000483, AvgCost = 212,1760, Sent = 1000, SentPerMin = 963,96, WordPerSec = 884,69
Update = 15500, Epoch = 16, LR = 0,000482, AvgCost = 219,0181, Sent = 265, SentPerMin = 945,94, WordPerSec = 877,88
Update = 15600, Epoch = 16, LR = 0,000480, AvgCost = 213,4016, Sent = 566, SentPerMin = 956,27, WordPerSec = 880,13
Update = 15700, Epoch = 16, LR = 0,000479, AvgCost = 212,7908, Sent = 859, SentPerMin = 957,67, WordPerSec = 882,49
Update = 15746, Epoch = 16, LR = 0,000478, AvgCost = 211,3131, Sent = 1000, SentPerMin = 965,95, WordPerSec = 886,52
Update = 15800, Epoch = 17, LR = 0,000477, AvgCost = 213,8995, Sent = 157, SentPerMin = 948,51, WordPerSec = 862,82
Update = 15900, Epoch = 17, LR = 0,000476, AvgCost = 214,9625, Sent = 450, SentPerMin = 950,07, WordPerSec = 876,07
Update = 16000, Epoch = 17, LR = 0,000474, AvgCost = 210,5428, Sent = 752, SentPerMin = 959,96, WordPerSec = 880,12
Update = 16084, Epoch = 17, LR = 0,000473, AvgCost = 210,4504, Sent = 1000, SentPerMin = 965,07, WordPerSec = 885,71
Update = 16100, Epoch = 18, LR = 0,000473, AvgCost = 205,9780, Sent = 51, SentPerMin = 1000,49, WordPerSec = 862,51
Update = 16200, Epoch = 18, LR = 0,000471, AvgCost = 216,2019, Sent = 336, SentPerMin = 946,00, WordPerSec = 876,46
Update = 16300, Epoch = 18, LR = 0,000470, AvgCost = 210,2729, Sent = 636, SentPerMin = 959,01, WordPerSec = 878,67
Update = 16400, Epoch = 18, LR = 0,000469, AvgCost = 210,1566, Sent = 935, SentPerMin = 963,91, WordPerSec = 886,08
Update = 16422, Epoch = 18, LR = 0,000468, AvgCost = 209,6937, Sent = 1000, SentPerMin = 967,71, WordPerSec = 888,13
Update = 16500, Epoch = 19, LR = 0,000467, AvgCost = 213,7211, Sent = 225, SentPerMin = 953,91, WordPerSec = 876,89
Update = 16600, Epoch = 19, LR = 0,000466, AvgCost = 212,1818, Sent = 524, SentPerMin = 952,14, WordPerSec = 876,09
Update = 16700, Epoch = 19, LR = 0,000464, AvgCost = 210,3840, Sent = 819, SentPerMin = 958,69, WordPerSec = 881,80
Update = 16760, Epoch = 19, LR = 0,000463, AvgCost = 209,0877, Sent = 1000, SentPerMin = 964,34, WordPerSec = 885,04
Update = 16800, Epoch = 20, LR = 0,000463, AvgCost = 208,3867, Sent = 122, SentPerMin = 708,29, WordPerSec = 618,21
Update = 16900, Epoch = 20, LR = 0,000462, AvgCost = 213,3499, Sent = 411, SentPerMin = 668,69, WordPerSec = 615,87
Update = 17000, Epoch = 20, LR = 0,000460, AvgCost = 209,4692, Sent = 709, SentPerMin = 677,09, WordPerSec = 618,48
Update = 17098, Epoch = 20, LR = 0,000459, AvgCost = 209,9481, Sent = 1000, SentPerMin = 676,04, WordPerSec = 620,45
Update = 17100, Epoch = 21, LR = 0,000459, AvgCost = 188,0454, Sent = 6, SentPerMin = 893,81, WordPerSec = 811,87
Update = 17200, Epoch = 21, LR = 0,000457, AvgCost = 218,1991, Sent = 292, SentPerMin = 936,21, WordPerSec = 875,94
Update = 17300, Epoch = 21, LR = 0,000456, AvgCost = 210,8946, Sent = 595, SentPerMin = 954,66, WordPerSec = 878,66
Update = 17400, Epoch = 21, LR = 0,000455, AvgCost = 210,1844, Sent = 890, SentPerMin = 958,52, WordPerSec = 883,58
Update = 17436, Epoch = 21, LR = 0,000454, AvgCost = 208,7043, Sent = 1000, SentPerMin = 964,95, WordPerSec = 885,60
Update = 17500, Epoch = 22, LR = 0,000454, AvgCost = 215,0371, Sent = 184, SentPerMin = 946,21, WordPerSec = 869,33
Update = 17600, Epoch = 22, LR = 0,000452, AvgCost = 210,6321, Sent = 482, SentPerMin = 954,41, WordPerSec = 877,45
Update = 17700, Epoch = 22, LR = 0,000451, AvgCost = 210,9350, Sent = 775, SentPerMin = 950,56, WordPerSec = 880,71
Update = 17774, Epoch = 22, LR = 0,000450, AvgCost = 207,6796, Sent = 1000, SentPerMin = 965,02, WordPerSec = 885,67
Update = 17800, Epoch = 23, LR = 0,000450, AvgCost = 202,9924, Sent = 81, SentPerMin = 969,54, WordPerSec = 860,61
Update = 17900, Epoch = 23, LR = 0,000448, AvgCost = 211,8782, Sent = 368, SentPerMin = 944,60, WordPerSec = 874,18
Update = 18000, Epoch = 23, LR = 0,000447, AvgCost = 208,0359, Sent = 663, SentPerMin = 957,79, WordPerSec = 880,00
Update = 18100, Epoch = 23, LR = 0,000446, AvgCost = 205,8612, Sent = 969, SentPerMin = 966,94, WordPerSec = 884,20
Update = 18112, Epoch = 23, LR = 0,000446, AvgCost = 206,7524, Sent = 1000, SentPerMin = 964,29, WordPerSec = 884,99
Update = 18200, Epoch = 24, LR = 0,000445, AvgCost = 216,1621, Sent = 251, SentPerMin = 934,91, WordPerSec = 873,39
Update = 18300, Epoch = 24, LR = 0,000444, AvgCost = 206,8899, Sent = 556, SentPerMin = 957,29, WordPerSec = 876,22
Update = 18400, Epoch = 24, LR = 0,000442, AvgCost = 207,3807, Sent = 847, SentPerMin = 955,67, WordPerSec = 880,84
Update = 18450, Epoch = 24, LR = 0,000442, AvgCost = 205,9483, Sent = 1000, SentPerMin = 963,94, WordPerSec = 884,68
Update = 18500, Epoch = 25, LR = 0,000441, AvgCost = 208,3669, Sent = 145, SentPerMin = 941,48, WordPerSec = 856,32
Update = 18600, Epoch = 25, LR = 0,000440, AvgCost = 210,5425, Sent = 437, SentPerMin = 947,17, WordPerSec = 877,74
Update = 18700, Epoch = 25, LR = 0,000439, AvgCost = 205,0116, Sent = 739, SentPerMin = 961,19, WordPerSec = 881,11
Update = 18788, Epoch = 25, LR = 0,000438, AvgCost = 205,0889, Sent = 1000, SentPerMin = 966,51, WordPerSec = 887,03
Update = 18800, Epoch = 26, LR = 0,000438, AvgCost = 166,9134, Sent = 41, SentPerMin = 1055,17, WordPerSec = 812,40
Update = 18900, Epoch = 26, LR = 0,000436, AvgCost = 210,5917, Sent = 324, SentPerMin = 941,39, WordPerSec = 874,41
Update = 19000, Epoch = 26, LR = 0,000435, AvgCost = 205,3171, Sent = 624, SentPerMin = 952,82, WordPerSec = 873,50
Update = 19100, Epoch = 26, LR = 0,000434, AvgCost = 204,9416, Sent = 921, SentPerMin = 958,58, WordPerSec = 881,94
Update = 19126, Epoch = 26, LR = 0,000434, AvgCost = 204,1850, Sent = 1000, SentPerMin = 962,61, WordPerSec = 883,45
Update = 19200, Epoch = 27, LR = 0,000433, AvgCost = 209,1127, Sent = 213, SentPerMin = 942,49, WordPerSec = 869,48
Update = 19300, Epoch = 27, LR = 0,000432, AvgCost = 207,2302, Sent = 511, SentPerMin = 946,64, WordPerSec = 872,91
Update = 19400, Epoch = 27, LR = 0,000431, AvgCost = 205,4440, Sent = 808, SentPerMin = 956,43, WordPerSec = 880,99
Update = 19464, Epoch = 27, LR = 0,000430, AvgCost = 203,6113, Sent = 1000, SentPerMin = 963,83, WordPerSec = 884,57
Update = 19500, Epoch = 28, LR = 0,000430, AvgCost = 193,3472, Sent = 113, SentPerMin = 1001,63, WordPerSec = 858,03
Update = 19600, Epoch = 28, LR = 0,000429, AvgCost = 206,0888, Sent = 399, SentPerMin = 942,00, WordPerSec = 867,08
Update = 19700, Epoch = 28, LR = 0,000427, AvgCost = 203,7320, Sent = 696, SentPerMin = 957,07, WordPerSec = 876,88
Update = 19800, Epoch = 28, LR = 0,000426, AvgCost = 203,2255, Sent = 995, SentPerMin = 962,70, WordPerSec = 882,92
Update = 19802, Epoch = 28, LR = 0,000426, AvgCost = 203,1846, Sent = 1000, SentPerMin = 962,14, WordPerSec = 883,02
Update = 19900, Epoch = 29, LR = 0,000425, AvgCost = 212,3343, Sent = 280, SentPerMin = 932,05, WordPerSec = 874,58
Update = 20000, Epoch = 29, LR = 0,000424, AvgCost = 203,2607, Sent = 588, SentPerMin = 958,84, WordPerSec = 877,09
Update = 20100, Epoch = 29, LR = 0,000423, AvgCost = 203,8412, Sent = 877, SentPerMin = 667,48, WordPerSec = 615,59
Update = 20140, Epoch = 29, LR = 0,000423, AvgCost = 202,5835, Sent = 1000, SentPerMin = 696,92, WordPerSec = 639,61
Update = 20200, Epoch = 30, LR = 0,000422, AvgCost = 207,9554, Sent = 173, SentPerMin = 932,23, WordPerSec = 857,69
Update = 20300, Epoch = 30, LR = 0,000421, AvgCost = 209,5036, Sent = 469, SentPerMin = 880,92, WordPerSec = 812,58
Update = 20400, Epoch = 30, LR = 0,000420, AvgCost = 207,2708, Sent = 766, SentPerMin = 789,94, WordPerSec = 728,91
Update = 20478, Epoch = 30, LR = 0,000419, AvgCost = 204,8342, Sent = 1000, SentPerMin = 768,23, WordPerSec = 705,06
Update = 20500, Epoch = 31, LR = 0,000419, AvgCost = 197,5005, Sent = 71, SentPerMin = 983,70, WordPerSec = 851,38
Update = 20600, Epoch = 31, LR = 0,000418, AvgCost = 208,6090, Sent = 358, SentPerMin = 943,89, WordPerSec = 867,03
Update = 20700, Epoch = 31, LR = 0,000417, AvgCost = 205,2147, Sent = 652, SentPerMin = 950,47, WordPerSec = 872,80
Update = 20800, Epoch = 31, LR = 0,000416, AvgCost = 203,5772, Sent = 955, SentPerMin = 957,61, WordPerSec = 878,55
Update = 20816, Epoch = 31, LR = 0,000416, AvgCost = 203,5670, Sent = 1000, SentPerMin = 957,64, WordPerSec = 878,89
Update = 20900, Epoch = 32, LR = 0,000415, AvgCost = 206,3114, Sent = 244, SentPerMin = 959,88, WordPerSec = 876,94
Update = 21000, Epoch = 32, LR = 0,000414, AvgCost = 205,4906, Sent = 542, SentPerMin = 955,72, WordPerSec = 877,78
Update = 21100, Epoch = 32, LR = 0,000413, AvgCost = 204,0190, Sent = 835, SentPerMin = 956,62, WordPerSec = 880,11
Update = 21154, Epoch = 32, LR = 0,000413, AvgCost = 202,7521, Sent = 1000, SentPerMin = 961,33, WordPerSec = 882,27
Update = 21200, Epoch = 33, LR = 0,000412, AvgCost = 201,9684, Sent = 136, SentPerMin = 963,56, WordPerSec = 857,40
Update = 21300, Epoch = 33, LR = 0,000411, AvgCost = 206,2717, Sent = 428, SentPerMin = 947,93, WordPerSec = 871,89
Update = 21400, Epoch = 33, LR = 0,000410, AvgCost = 202,6653, Sent = 725, SentPerMin = 958,20, WordPerSec = 878,35
Update = 21492, Epoch = 33, LR = 0,000409, AvgCost = 202,1004, Sent = 1000, SentPerMin = 961,87, WordPerSec = 882,78
Update = 21500, Epoch = 34, LR = 0,000409, AvgCost = 185,0346, Sent = 23, SentPerMin = 930,21, WordPerSec = 810,90
Update = 21600, Epoch = 34, LR = 0,000408, AvgCost = 209,4820, Sent = 311, SentPerMin = 937,42, WordPerSec = 871,56
Update = 21700, Epoch = 34, LR = 0,000407, AvgCost = 204,0849, Sent = 611, SentPerMin = 946,36, WordPerSec = 871,63
Update = 21800, Epoch = 34, LR = 0,000406, AvgCost = 202,7583, Sent = 906, SentPerMin = 951,89, WordPerSec = 878,11
Update = 21830, Epoch = 34, LR = 0,000406, AvgCost = 201,3558, Sent = 1000, SentPerMin = 959,14, WordPerSec = 880,27
Update = 21900, Epoch = 35, LR = 0,000405, AvgCost = 206,7805, Sent = 200, SentPerMin = 938,18, WordPerSec = 860,93
Update = 22000, Epoch = 35, LR = 0,000405, AvgCost = 203,7333, Sent = 501, SentPerMin = 946,36, WordPerSec = 866,99
Update = 22100, Epoch = 35, LR = 0,000404, AvgCost = 202,9952, Sent = 795, SentPerMin = 945,54, WordPerSec = 871,23
Update = 22168, Epoch = 35, LR = 0,000403, AvgCost = 200,8994, Sent = 1000, SentPerMin = 955,30, WordPerSec = 876,74
Update = 22200, Epoch = 36, LR = 0,000403, AvgCost = 196,5579, Sent = 99, SentPerMin = 984,69, WordPerSec = 858,04
Update = 22300, Epoch = 36, LR = 0,000402, AvgCost = 204,2755, Sent = 388, SentPerMin = 949,28, WordPerSec = 870,06
Update = 22400, Epoch = 36, LR = 0,000401, AvgCost = 201,5206, Sent = 683, SentPerMin = 955,89, WordPerSec = 875,88
Update = 22500, Epoch = 36, LR = 0,000400, AvgCost = 200,6271, Sent = 984, SentPerMin = 960,65, WordPerSec = 881,58
Update = 22506, Epoch = 36, LR = 0,000400, AvgCost = 200,5045, Sent = 1000, SentPerMin = 960,23, WordPerSec = 881,27
Update = 22600, Epoch = 37, LR = 0,000399, AvgCost = 209,6576, Sent = 269, SentPerMin = 933,71, WordPerSec = 872,74
Update = 22700, Epoch = 37, LR = 0,000398, AvgCost = 202,4463, Sent = 573, SentPerMin = 952,90, WordPerSec = 875,16
Update = 22800, Epoch = 37, LR = 0,000397, AvgCost = 200,9662, Sent = 865, SentPerMin = 954,35, WordPerSec = 878,99
Update = 22844, Epoch = 37, LR = 0,000397, AvgCost = 199,9354, Sent = 1000, SentPerMin = 961,06, WordPerSec = 882,03
Update = 22900, Epoch = 38, LR = 0,000396, AvgCost = 203,9918, Sent = 161, SentPerMin = 932,11, WordPerSec = 855,02
Update = 23000, Epoch = 38, LR = 0,000396, AvgCost = 205,0664, Sent = 456, SentPerMin = 941,56, WordPerSec = 869,98
Update = 23100, Epoch = 38, LR = 0,000395, AvgCost = 200,6200, Sent = 757, SentPerMin = 950,67, WordPerSec = 874,44
Update = 23182, Epoch = 38, LR = 0,000394, AvgCost = 199,2952, Sent = 1000, SentPerMin = 959,82, WordPerSec = 880,89
Update = 23200, Epoch = 39, LR = 0,000394, AvgCost = 189,6985, Sent = 59, SentPerMin = 739,38, WordPerSec = 617,82
Update = 23300, Epoch = 39, LR = 0,000393, AvgCost = 205,9054, Sent = 343, SentPerMin = 669,32, WordPerSec = 618,85
Update = 23400, Epoch = 39, LR = 0,000392, AvgCost = 200,9434, Sent = 640, SentPerMin = 757,56, WordPerSec = 696,14
Update = 23500, Epoch = 39, LR = 0,000391, AvgCost = 199,0820, Sent = 943, SentPerMin = 811,05, WordPerSec = 744,45
Update = 23520, Epoch = 39, LR = 0,000391, AvgCost = 199,0253, Sent = 1000, SentPerMin = 818,16, WordPerSec = 750,88
Update = 23600, Epoch = 40, LR = 0,000391, AvgCost = 202,6082, Sent = 231, SentPerMin = 927,54, WordPerSec = 853,53
Update = 23700, Epoch = 40, LR = 0,000390, AvgCost = 201,7603, Sent = 529, SentPerMin = 840,51, WordPerSec = 773,59
Update = 23800, Epoch = 40, LR = 0,000389, AvgCost = 200,1672, Sent = 824, SentPerMin = 775,04, WordPerSec = 713,18
Update = 23858, Epoch = 40, LR = 0,000388, AvgCost = 199,0349, Sent = 1000, SentPerMin = 760,43, WordPerSec = 697,90
Update = 23900, Epoch = 41, LR = 0,000388, AvgCost = 197,2221, Sent = 126, SentPerMin = 966,37, WordPerSec = 853,12
Update = 24000, Epoch = 41, LR = 0,000387, AvgCost = 200,8330, Sent = 418, SentPerMin = 945,21, WordPerSec = 867,35
Update = 24100, Epoch = 41, LR = 0,000386, AvgCost = 197,6977, Sent = 715, SentPerMin = 957,37, WordPerSec = 875,52
Update = 24196, Epoch = 41, LR = 0,000386, AvgCost = 197,8865, Sent = 1000, SentPerMin = 958,54, WordPerSec = 879,71
Update = 24200, Epoch = 42, LR = 0,000386, AvgCost = 178,7953, Sent = 12, SentPerMin = 969,35, WordPerSec = 872,42
Update = 24300, Epoch = 42, LR = 0,000385, AvgCost = 205,6803, Sent = 299, SentPerMin = 933,43, WordPerSec = 870,68
Update = 24400, Epoch = 42, LR = 0,000384, AvgCost = 199,0038, Sent = 601, SentPerMin = 949,00, WordPerSec = 871,78
Update = 24500, Epoch = 42, LR = 0,000383, AvgCost = 198,0469, Sent = 897, SentPerMin = 953,38, WordPerSec = 876,95
Update = 24534, Epoch = 42, LR = 0,000383, AvgCost = 197,2604, Sent = 1000, SentPerMin = 958,17, WordPerSec = 879,37
Update = 24600, Epoch = 43, LR = 0,000383, AvgCost = 203,9099, Sent = 189, SentPerMin = 938,64, WordPerSec = 864,64
Update = 24700, Epoch = 43, LR = 0,000382, AvgCost = 198,9311, Sent = 490, SentPerMin = 952,60, WordPerSec = 870,85
Update = 24800, Epoch = 43, LR = 0,000381, AvgCost = 198,8725, Sent = 784, SentPerMin = 948,70, WordPerSec = 874,34
Update = 24872, Epoch = 43, LR = 0,000380, AvgCost = 196,6164, Sent = 1000, SentPerMin = 958,55, WordPerSec = 879,72
Update = 24900, Epoch = 44, LR = 0,000380, AvgCost = 191,2763, Sent = 87, SentPerMin = 980,04, WordPerSec = 861,19
Update = 25000, Epoch = 44, LR = 0,000379, AvgCost = 200,2858, Sent = 375, SentPerMin = 939,52, WordPerSec = 865,95
Update = 25100, Epoch = 44, LR = 0,000379, AvgCost = 197,8401, Sent = 668, SentPerMin = 948,66, WordPerSec = 874,17
Update = 25200, Epoch = 44, LR = 0,000378, AvgCost = 195,0030, Sent = 974, SentPerMin = 961,56, WordPerSec = 879,90
Update = 25210, Epoch = 44, LR = 0,000378, AvgCost = 195,9983, Sent = 1000, SentPerMin = 959,23, WordPerSec = 880,35
Update = 25300, Epoch = 45, LR = 0,000377, AvgCost = 204,1073, Sent = 258, SentPerMin = 934,99, WordPerSec = 871,51
Update = 25400, Epoch = 45, LR = 0,000376, AvgCost = 196,6672, Sent = 562, SentPerMin = 955,95, WordPerSec = 876,55
Update = 25500, Epoch = 45, LR = 0,000376, AvgCost = 196,5290, Sent = 854, SentPerMin = 954,91, WordPerSec = 879,36
Update = 25548, Epoch = 45, LR = 0,000375, AvgCost = 195,3816, Sent = 1000, SentPerMin = 961,29, WordPerSec = 882,24
Update = 25600, Epoch = 46, LR = 0,000375, AvgCost = 195,2900, Sent = 152, SentPerMin = 943,74, WordPerSec = 849,06
Update = 25700, Epoch = 46, LR = 0,000374, AvgCost = 200,1010, Sent = 443, SentPerMin = 941,11, WordPerSec = 870,12
Update = 25800, Epoch = 46, LR = 0,000374, AvgCost = 195,4330, Sent = 744, SentPerMin = 950,98, WordPerSec = 873,41
Update = 25886, Epoch = 46, LR = 0,000373, AvgCost = 194,8845, Sent = 1000, SentPerMin = 958,38, WordPerSec = 879,57
Update = 25900, Epoch = 47, LR = 0,000373, AvgCost = 182,9841, Sent = 45, SentPerMin = 1017,08, WordPerSec = 832,88
Update = 26000, Epoch = 47, LR = 0,000372, AvgCost = 200,7274, Sent = 330, SentPerMin = 934,30, WordPerSec = 866,06
Update = 26100, Epoch = 47, LR = 0,000371, AvgCost = 195,7994, Sent = 630, SentPerMin = 947,50, WordPerSec = 868,92
Update = 26200, Epoch = 47, LR = 0,000371, AvgCost = 194,8712, Sent = 929, SentPerMin = 954,28, WordPerSec = 876,74
Update = 26224, Epoch = 47, LR = 0,000371, AvgCost = 194,4506, Sent = 1000, SentPerMin = 957,42, WordPerSec = 878,69
Update = 26300, Epoch = 48, LR = 0,000370, AvgCost = 199,3192, Sent = 219, SentPerMin = 932,58, WordPerSec = 859,33
Update = 26400, Epoch = 48, LR = 0,000369, AvgCost = 197,0014, Sent = 519, SentPerMin = 947,20, WordPerSec = 869,18
Update = 26500, Epoch = 48, LR = 0,000369, AvgCost = 195,5192, Sent = 814, SentPerMin = 950,22, WordPerSec = 874,01
Update = 26562, Epoch = 48, LR = 0,000368, AvgCost = 194,1387, Sent = 1000, SentPerMin = 956,57, WordPerSec = 877,90
Update = 26600, Epoch = 49, LR = 0,000368, AvgCost = 182,3972, Sent = 119, SentPerMin = 723,02, WordPerSec = 615,07
Update = 26700, Epoch = 49, LR = 0,000367, AvgCost = 198,3574, Sent = 406, SentPerMin = 683,02, WordPerSec = 627,76
Update = 26800, Epoch = 49, LR = 0,000367, AvgCost = 194,9124, Sent = 702, SentPerMin = 774,94, WordPerSec = 708,70
Update = 26900, Epoch = 49, LR = 0,000366, AvgCost = 194,4797, Sent = 1000, SentPerMin = 818,87, WordPerSec = 751,53
Update = 26900, Epoch = 49, LR = 0,000366, AvgCost = 194,4797, Sent = 1000, SentPerMin = 818,86, WordPerSec = 751,53
Update = 27000, Epoch = 50, LR = 0,000365, AvgCost = 204,2089, Sent = 285, SentPerMin = 906,85, WordPerSec = 851,54
Update = 27100, Epoch = 50, LR = 0,000364, AvgCost = 195,5096, Sent = 592, SentPerMin = 824,13, WordPerSec = 755,99
Update = 27200, Epoch = 50, LR = 0,000364, AvgCost = 196,3360, Sent = 882, SentPerMin = 768,74, WordPerSec = 710,19
Update = 27238, Epoch = 50, LR = 0,000364, AvgCost = 194,6705, Sent = 1000, SentPerMin = 762,77, WordPerSec = 700,05
Update = 27300, Epoch = 51, LR = 0,000363, AvgCost = 197,2773, Sent = 178, SentPerMin = 933,66, WordPerSec = 861,54
Update = 27400, Epoch = 51, LR = 0,000362, AvgCost = 197,6257, Sent = 475, SentPerMin = 941,96, WordPerSec = 867,66
Update = 27500, Epoch = 51, LR = 0,000362, AvgCost = 196,2588, Sent = 771, SentPerMin = 946,10, WordPerSec = 874,48
Update = 27576, Epoch = 51, LR = 0,000361, AvgCost = 193,7385, Sent = 1000, SentPerMin = 958,25, WordPerSec = 879,45
Update = 27600, Epoch = 52, LR = 0,000361, AvgCost = 188,6908, Sent = 76, SentPerMin = 974,48, WordPerSec = 859,51
Update = 27700, Epoch = 52, LR = 0,000361, AvgCost = 197,5466, Sent = 363, SentPerMin = 944,77, WordPerSec = 871,20
Update = 27800, Epoch = 52, LR = 0,000360, AvgCost = 194,7223, Sent = 656, SentPerMin = 950,09, WordPerSec = 874,78
Update = 27900, Epoch = 52, LR = 0,000359, AvgCost = 192,7303, Sent = 961, SentPerMin = 960,88, WordPerSec = 880,71
Update = 27914, Epoch = 52, LR = 0,000359, AvgCost = 193,0713, Sent = 1000, SentPerMin = 960,50, WordPerSec = 881,51
Update = 28000, Epoch = 53, LR = 0,000359, AvgCost = 197,5229, Sent = 248, SentPerMin = 940,16, WordPerSec = 868,51
Update = 28100, Epoch = 53, LR = 0,000358, AvgCost = 194,1017, Sent = 549, SentPerMin = 952,17, WordPerSec = 872,42
Update = 28200, Epoch = 53, LR = 0,000357, AvgCost = 193,9006, Sent = 841, SentPerMin = 949,95, WordPerSec = 874,76
Update = 28252, Epoch = 53, LR = 0,000357, AvgCost = 192,5663, Sent = 1000, SentPerMin = 956,02, WordPerSec = 877,40
Update = 28300, Epoch = 54, LR = 0,000357, AvgCost = 191,3851, Sent = 140, SentPerMin = 954,65, WordPerSec = 856,91
Update = 28400, Epoch = 54, LR = 0,000356, AvgCost = 198,2830, Sent = 431, SentPerMin = 939,59, WordPerSec = 870,96
Update = 28500, Epoch = 54, LR = 0,000355, AvgCost = 192,1303, Sent = 732, SentPerMin = 958,76, WordPerSec = 876,55
Update = 28590, Epoch = 54, LR = 0,000355, AvgCost = 192,2606, Sent = 1000, SentPerMin = 959,00, WordPerSec = 880,14
Update = 28600, Epoch = 55, LR = 0,000355, AvgCost = 158,6068, Sent = 33, SentPerMin = 1053,04, WordPerSec = 813,71
Update = 28700, Epoch = 55, LR = 0,000354, AvgCost = 199,2335, Sent = 317, SentPerMin = 932,05, WordPerSec = 867,95
Update = 28800, Epoch = 55, LR = 0,000354, AvgCost = 193,5681, Sent = 618, SentPerMin = 947,03, WordPerSec = 869,15
Update = 28900, Epoch = 55, LR = 0,000353, AvgCost = 192,9664, Sent = 914, SentPerMin = 951,08, WordPerSec = 876,42
Update = 28928, Epoch = 55, LR = 0,000353, AvgCost = 191,8755, Sent = 1000, SentPerMin = 956,88, WordPerSec = 878,20
Update = 29000, Epoch = 56, LR = 0,000352, AvgCost = 199,8821, Sent = 205, SentPerMin = 929,42, WordPerSec = 867,76
Update = 29100, Epoch = 56, LR = 0,000352, AvgCost = 193,8623, Sent = 506, SentPerMin = 946,97, WordPerSec = 868,59
Update = 29200, Epoch = 56, LR = 0,000351, AvgCost = 193,4774, Sent = 802, SentPerMin = 950,14, WordPerSec = 874,74
Update = 29266, Epoch = 56, LR = 0,000351, AvgCost = 191,6712, Sent = 1000, SentPerMin = 957,93, WordPerSec = 879,16
Update = 29300, Epoch = 57, LR = 0,000351, AvgCost = 183,1913, Sent = 106, SentPerMin = 983,55, WordPerSec = 847,15
Update = 29400, Epoch = 57, LR = 0,000350, AvgCost = 195,3875, Sent = 393, SentPerMin = 940,37, WordPerSec = 866,67
Update = 29500, Epoch = 57, LR = 0,000349, AvgCost = 192,2048, Sent = 689, SentPerMin = 955,50, WordPerSec = 875,87
Update = 29600, Epoch = 57, LR = 0,000349, AvgCost = 191,5045, Sent = 989, SentPerMin = 958,51, WordPerSec = 879,92
Update = 29604, Epoch = 57, LR = 0,000349, AvgCost = 191,3523, Sent = 1000, SentPerMin = 958,59, WordPerSec = 879,76
Update = 29700, Epoch = 58, LR = 0,000348, AvgCost = 201,5269, Sent = 275, SentPerMin = 922,90, WordPerSec = 866,35
Update = 29800, Epoch = 58, LR = 0,000348, AvgCost = 192,5054, Sent = 581, SentPerMin = 948,49, WordPerSec = 870,11
Update = 29900, Epoch = 58, LR = 0,000347, AvgCost = 192,2651, Sent = 870, SentPerMin = 948,21, WordPerSec = 875,46
Update = 29942, Epoch = 58, LR = 0,000347, AvgCost = 190,7517, Sent = 1000, SentPerMin = 956,12, WordPerSec = 877,50
Update = 30000, Epoch = 59, LR = 0,000346, AvgCost = 196,7527, Sent = 166, SentPerMin = 667,20, WordPerSec = 614,81
Update = 30100, Epoch = 59, LR = 0,000346, AvgCost = 198,1659, Sent = 463, SentPerMin = 576,55, WordPerSec = 532,15
Update = 30200, Epoch = 59, LR = 0,000345, AvgCost = 194,9153, Sent = 761, SentPerMin = 683,81, WordPerSec = 629,69
Update = 30280, Epoch = 59, LR = 0,000345, AvgCost = 192,3476, Sent = 1000, SentPerMin = 736,73, WordPerSec = 676,14
Update = 30300, Epoch = 60, LR = 0,000345, AvgCost = 187,1740, Sent = 64, SentPerMin = 990,22, WordPerSec = 863,35
Update = 30400, Epoch = 60, LR = 0,000344, AvgCost = 196,9929, Sent = 352, SentPerMin = 945,63, WordPerSec = 869,61
Update = 30500, Epoch = 60, LR = 0,000344, AvgCost = 194,3101, Sent = 645, SentPerMin = 812,35, WordPerSec = 747,74
Update = 30600, Epoch = 60, LR = 0,000343, AvgCost = 193,0703, Sent = 948, SentPerMin = 768,83, WordPerSec = 706,79
Update = 30618, Epoch = 60, LR = 0,000343, AvgCost = 192,6095, Sent = 1000, SentPerMin = 764,08, WordPerSec = 701,25
Update = 30700, Epoch = 61, LR = 0,000342, AvgCost = 194,9122, Sent = 239, SentPerMin = 959,95, WordPerSec = 876,68
Update = 30800, Epoch = 61, LR = 0,000342, AvgCost = 194,7267, Sent = 536, SentPerMin = 952,19, WordPerSec = 875,24
Update = 30900, Epoch = 61, LR = 0,000341, AvgCost = 192,5580, Sent = 831, SentPerMin = 955,69, WordPerSec = 877,54
Update = 30956, Epoch = 61, LR = 0,000341, AvgCost = 191,7211, Sent = 1000, SentPerMin = 961,08, WordPerSec = 882,04
Update = 31000, Epoch = 62, LR = 0,000341, AvgCost = 190,3522, Sent = 131, SentPerMin = 958,64, WordPerSec = 853,14
Update = 31100, Epoch = 62, LR = 0,000340, AvgCost = 194,2793, Sent = 423, SentPerMin = 944,88, WordPerSec = 868,33
Update = 31200, Epoch = 62, LR = 0,000340, AvgCost = 191,2737, Sent = 720, SentPerMin = 956,75, WordPerSec = 875,58
Update = 31294, Epoch = 62, LR = 0,000339, AvgCost = 191,0523, Sent = 1000, SentPerMin = 958,63, WordPerSec = 879,80
Update = 31300, Epoch = 63, LR = 0,000339, AvgCost = 193,0632, Sent = 16, SentPerMin = 858,63, WordPerSec = 838,06
Update = 31400, Epoch = 63, LR = 0,000339, AvgCost = 199,4068, Sent = 304, SentPerMin = 934,00, WordPerSec = 872,35
Update = 31500, Epoch = 63, LR = 0,000338, AvgCost = 192,9378, Sent = 606, SentPerMin = 947,37, WordPerSec = 870,69
Update = 31600, Epoch = 63, LR = 0,000338, AvgCost = 192,0932, Sent = 902, SentPerMin = 952,15, WordPerSec = 876,92
Update = 31632, Epoch = 63, LR = 0,000337, AvgCost = 190,8631, Sent = 1000, SentPerMin = 957,97, WordPerSec = 879,20
Update = 31700, Epoch = 64, LR = 0,000337, AvgCost = 196,5102, Sent = 195, SentPerMin = 937,02, WordPerSec = 861,26
Update = 31800, Epoch = 64, LR = 0,000336, AvgCost = 192,8531, Sent = 495, SentPerMin = 952,60, WordPerSec = 870,58
Update = 31900, Epoch = 64, LR = 0,000336, AvgCost = 192,8367, Sent = 789, SentPerMin = 949,27, WordPerSec = 875,30
Update = 31970, Epoch = 64, LR = 0,000336, AvgCost = 190,4509, Sent = 1000, SentPerMin = 959,09, WordPerSec = 880,22
Update = 32000, Epoch = 65, LR = 0,000335, AvgCost = 188,8931, Sent = 92, SentPerMin = 959,26, WordPerSec = 851,34
Update = 32100, Epoch = 65, LR = 0,000335, AvgCost = 194,5933, Sent = 381, SentPerMin = 942,09, WordPerSec = 866,92
Update = 32200, Epoch = 65, LR = 0,000334, AvgCost = 191,7201, Sent = 676, SentPerMin = 953,83, WordPerSec = 875,68
Update = 32300, Epoch = 65, LR = 0,000334, AvgCost = 189,3606, Sent = 979, SentPerMin = 960,88, WordPerSec = 879,84
Update = 32308, Epoch = 65, LR = 0,000334, AvgCost = 190,2574, Sent = 1000, SentPerMin = 958,79, WordPerSec = 879,95
Starting inference...
Inference results:
和在在了了了了了了了了了"。
在在在•。的了了。
和在在了•••••••••••和和和了了•了了和了了了了和•••••••••和和和和了了了了了•••的••••••和和和和了
在在在了。
在在了了了了。

             */


            /*
             Update = 10100, Epoch = 0, LR = 0,000597, AvgCost = 266,7342, Sent = 285, SentPerMin = 623,31, WordPerSec = 585,29
Update = 10200, Epoch = 0, LR = 0,000594, AvgCost = 248,1456, Sent = 592, SentPerMin = 657,97, WordPerSec = 603,57
Update = 10300, Epoch = 0, LR = 0,000591, AvgCost = 245,8514, Sent = 882, SentPerMin = 662,94, WordPerSec = 612,44
Update = 10338, Epoch = 0, LR = 0,000590, AvgCost = 242,6576, Sent = 1000, SentPerMin = 669,45, WordPerSec = 614,40
Update = 10400, Epoch = 1, LR = 0,000588, AvgCost = 232,8826, Sent = 178, SentPerMin = 935,20, WordPerSec = 862,95
Update = 10500, Epoch = 1, LR = 0,000586, AvgCost = 233,6964, Sent = 475, SentPerMin = 946,95, WordPerSec = 872,26
Update = 10600, Epoch = 1, LR = 0,000583, AvgCost = 232,3053, Sent = 771, SentPerMin = 948,33, WordPerSec = 876,54
Update = 10676, Epoch = 1, LR = 0,000581, AvgCost = 229,1750, Sent = 1000, SentPerMin = 960,35, WordPerSec = 881,37
Update = 10700, Epoch = 2, LR = 0,000580, AvgCost = 220,6970, Sent = 76, SentPerMin = 971,87, WordPerSec = 857,21
Update = 10800, Epoch = 2, LR = 0,000577, AvgCost = 230,5997, Sent = 363, SentPerMin = 935,44, WordPerSec = 862,59
Update = 10900, Epoch = 2, LR = 0,000575, AvgCost = 227,5202, Sent = 656, SentPerMin = 937,75, WordPerSec = 863,42
Update = 11000, Epoch = 2, LR = 0,000572, AvgCost = 225,0590, Sent = 961, SentPerMin = 950,22, WordPerSec = 870,94
Update = 11014, Epoch = 2, LR = 0,000572, AvgCost = 225,4237, Sent = 1000, SentPerMin = 949,24, WordPerSec = 871,18
Update = 11100, Epoch = 3, LR = 0,000569, AvgCost = 227,8687, Sent = 248, SentPerMin = 921,75, WordPerSec = 851,50
Update = 11200, Epoch = 3, LR = 0,000567, AvgCost = 224,4464, Sent = 549, SentPerMin = 931,20, WordPerSec = 853,20
Update = 11300, Epoch = 3, LR = 0,000564, AvgCost = 224,5554, Sent = 841, SentPerMin = 935,42, WordPerSec = 861,38
Update = 11352, Epoch = 3, LR = 0,000563, AvgCost = 223,1561, Sent = 1000, SentPerMin = 940,77, WordPerSec = 863,41
Update = 11400, Epoch = 4, LR = 0,000562, AvgCost = 220,8584, Sent = 140, SentPerMin = 948,21, WordPerSec = 851,13
Update = 11500, Epoch = 4, LR = 0,000560, AvgCost = 228,3223, Sent = 431, SentPerMin = 934,90, WordPerSec = 866,61
Update = 11600, Epoch = 4, LR = 0,000557, AvgCost = 221,4793, Sent = 732, SentPerMin = 947,15, WordPerSec = 865,93
Update = 11690, Epoch = 4, LR = 0,000555, AvgCost = 221,9501, Sent = 1000, SentPerMin = 951,53, WordPerSec = 873,28
Update = 11700, Epoch = 5, LR = 0,000555, AvgCost = 182,2629, Sent = 33, SentPerMin = 1044,28, WordPerSec = 806,94
Update = 11800, Epoch = 5, LR = 0,000552, AvgCost = 228,2567, Sent = 317, SentPerMin = 927,29, WordPerSec = 863,52
Update = 11900, Epoch = 5, LR = 0,000550, AvgCost = 222,1995, Sent = 618, SentPerMin = 945,68, WordPerSec = 867,92
Update = 12000, Epoch = 5, LR = 0,000548, AvgCost = 221,9816, Sent = 914, SentPerMin = 949,63, WordPerSec = 875,08
Update = 12028, Epoch = 5, LR = 0,000547, AvgCost = 220,7257, Sent = 1000, SentPerMin = 955,75, WordPerSec = 877,15
Update = 12100, Epoch = 6, LR = 0,000545, AvgCost = 227,9262, Sent = 205, SentPerMin = 923,14, WordPerSec = 861,90
Update = 12200, Epoch = 6, LR = 0,000543, AvgCost = 221,3272, Sent = 506, SentPerMin = 944,82, WordPerSec = 866,62
Update = 12300, Epoch = 6, LR = 0,000541, AvgCost = 221,9363, Sent = 802, SentPerMin = 948,22, WordPerSec = 872,96
Update = 12366, Epoch = 6, LR = 0,000540, AvgCost = 219,9903, Sent = 1000, SentPerMin = 954,97, WordPerSec = 876,44
Update = 12400, Epoch = 7, LR = 0,000539, AvgCost = 209,0330, Sent = 106, SentPerMin = 987,24, WordPerSec = 850,33
Update = 12500, Epoch = 7, LR = 0,000537, AvgCost = 222,5717, Sent = 393, SentPerMin = 943,30, WordPerSec = 869,37
Update = 12600, Epoch = 7, LR = 0,000535, AvgCost = 219,4649, Sent = 689, SentPerMin = 957,76, WordPerSec = 877,94
Update = 12700, Epoch = 7, LR = 0,000532, AvgCost = 219,3996, Sent = 989, SentPerMin = 958,76, WordPerSec = 880,15
Update = 12704, Epoch = 7, LR = 0,000532, AvgCost = 219,2368, Sent = 1000, SentPerMin = 958,89, WordPerSec = 880,03
Update = 12800, Epoch = 8, LR = 0,000530, AvgCost = 228,8796, Sent = 275, SentPerMin = 916,31, WordPerSec = 860,17
Update = 12900, Epoch = 8, LR = 0,000528, AvgCost = 219,2504, Sent = 581, SentPerMin = 943,94, WordPerSec = 865,93
Update = 13000, Epoch = 8, LR = 0,000526, AvgCost = 219,9270, Sent = 870, SentPerMin = 946,55, WordPerSec = 873,92
Update = 13042, Epoch = 8, LR = 0,000525, AvgCost = 218,4195, Sent = 1000, SentPerMin = 955,78, WordPerSec = 877,18
Update = 13100, Epoch = 9, LR = 0,000524, AvgCost = 221,1126, Sent = 166, SentPerMin = 925,81, WordPerSec = 853,12
Update = 13200, Epoch = 9, LR = 0,000522, AvgCost = 221,6514, Sent = 463, SentPerMin = 941,32, WordPerSec = 868,84
Update = 13300, Epoch = 9, LR = 0,000520, AvgCost = 219,6542, Sent = 761, SentPerMin = 948,76, WordPerSec = 873,66
Update = 13380, Epoch = 9, LR = 0,000519, AvgCost = 217,6848, Sent = 1000, SentPerMin = 957,22, WordPerSec = 878,50
Update = 13400, Epoch = 10, LR = 0,000518, AvgCost = 213,9592, Sent = 64, SentPerMin = 709,86, WordPerSec = 618,91
Update = 13500, Epoch = 10, LR = 0,000516, AvgCost = 221,9903, Sent = 352, SentPerMin = 670,44, WordPerSec = 616,54
Update = 13600, Epoch = 10, LR = 0,000514, AvgCost = 218,8846, Sent = 645, SentPerMin = 670,62, WordPerSec = 617,29
Update = 13700, Epoch = 10, LR = 0,000513, AvgCost = 218,0834, Sent = 948, SentPerMin = 675,08, WordPerSec = 620,60
Update = 13718, Epoch = 10, LR = 0,000512, AvgCost = 217,6140, Sent = 1000, SentPerMin = 676,23, WordPerSec = 620,62
Update = 13800, Epoch = 11, LR = 0,000511, AvgCost = 219,1710, Sent = 239, SentPerMin = 950,16, WordPerSec = 867,73
Update = 13900, Epoch = 11, LR = 0,000509, AvgCost = 218,7733, Sent = 536, SentPerMin = 947,00, WordPerSec = 870,46
Update = 14000, Epoch = 11, LR = 0,000507, AvgCost = 217,3611, Sent = 831, SentPerMin = 951,92, WordPerSec = 874,08
Update = 14056, Epoch = 11, LR = 0,000506, AvgCost = 216,8055, Sent = 1000, SentPerMin = 956,79, WordPerSec = 878,11
Update = 14100, Epoch = 12, LR = 0,000505, AvgCost = 215,7036, Sent = 131, SentPerMin = 950,39, WordPerSec = 845,80
Update = 14200, Epoch = 12, LR = 0,000504, AvgCost = 218,3424, Sent = 423, SentPerMin = 940,05, WordPerSec = 863,90
Update = 14300, Epoch = 12, LR = 0,000502, AvgCost = 215,5052, Sent = 720, SentPerMin = 955,43, WordPerSec = 874,37
Update = 14394, Epoch = 12, LR = 0,000500, AvgCost = 215,9168, Sent = 1000, SentPerMin = 956,18, WordPerSec = 877,55
Update = 14400, Epoch = 13, LR = 0,000500, AvgCost = 218,2275, Sent = 16, SentPerMin = 851,08, WordPerSec = 830,69
Update = 14500, Epoch = 13, LR = 0,000498, AvgCost = 223,0878, Sent = 304, SentPerMin = 934,34, WordPerSec = 872,67
Update = 14600, Epoch = 13, LR = 0,000497, AvgCost = 216,1755, Sent = 606, SentPerMin = 947,26, WordPerSec = 870,59
Update = 14700, Epoch = 13, LR = 0,000495, AvgCost = 215,9581, Sent = 902, SentPerMin = 951,22, WordPerSec = 876,07
Update = 14732, Epoch = 13, LR = 0,000494, AvgCost = 214,8380, Sent = 1000, SentPerMin = 956,47, WordPerSec = 877,82
Update = 14800, Epoch = 14, LR = 0,000493, AvgCost = 219,7715, Sent = 195, SentPerMin = 939,33, WordPerSec = 863,38
Update = 14900, Epoch = 14, LR = 0,000492, AvgCost = 215,2519, Sent = 495, SentPerMin = 955,81, WordPerSec = 873,52
Update = 15000, Epoch = 14, LR = 0,000490, AvgCost = 216,1024, Sent = 789, SentPerMin = 947,81, WordPerSec = 873,95
Update = 15070, Epoch = 14, LR = 0,000489, AvgCost = 213,9121, Sent = 1000, SentPerMin = 957,60, WordPerSec = 878,85
Update = 15100, Epoch = 15, LR = 0,000488, AvgCost = 211,8290, Sent = 92, SentPerMin = 966,60, WordPerSec = 857,86
Update = 15200, Epoch = 15, LR = 0,000487, AvgCost = 216,6044, Sent = 381, SentPerMin = 945,68, WordPerSec = 870,23
Update = 15300, Epoch = 15, LR = 0,000485, AvgCost = 214,1051, Sent = 676, SentPerMin = 953,79, WordPerSec = 875,65
Update = 15400, Epoch = 15, LR = 0,000483, AvgCost = 211,9148, Sent = 979, SentPerMin = 962,34, WordPerSec = 881,18
Update = 15408, Epoch = 15, LR = 0,000483, AvgCost = 213,0517, Sent = 1000, SentPerMin = 960,34, WordPerSec = 881,36
Update = 15500, Epoch = 16, LR = 0,000482, AvgCost = 220,3790, Sent = 265, SentPerMin = 941,06, WordPerSec = 873,35
Update = 15600, Epoch = 16, LR = 0,000480, AvgCost = 214,7029, Sent = 566, SentPerMin = 950,70, WordPerSec = 875,00
Update = 15700, Epoch = 16, LR = 0,000479, AvgCost = 213,7874, Sent = 859, SentPerMin = 951,42, WordPerSec = 876,73
Update = 15746, Epoch = 16, LR = 0,000478, AvgCost = 212,5351, Sent = 1000, SentPerMin = 958,64, WordPerSec = 879,81
Update = 15800, Epoch = 17, LR = 0,000477, AvgCost = 215,7353, Sent = 157, SentPerMin = 941,09, WordPerSec = 856,07
Update = 15900, Epoch = 17, LR = 0,000476, AvgCost = 216,7408, Sent = 450, SentPerMin = 941,00, WordPerSec = 867,71
Update = 16000, Epoch = 17, LR = 0,000474, AvgCost = 211,9948, Sent = 752, SentPerMin = 952,64, WordPerSec = 873,40
Update = 16084, Epoch = 17, LR = 0,000473, AvgCost = 212,0651, Sent = 1000, SentPerMin = 958,77, WordPerSec = 879,93
Update = 16100, Epoch = 18, LR = 0,000473, AvgCost = 206,7638, Sent = 51, SentPerMin = 998,63, WordPerSec = 860,91
Update = 16200, Epoch = 18, LR = 0,000471, AvgCost = 218,0378, Sent = 336, SentPerMin = 941,07, WordPerSec = 871,89
Update = 16300, Epoch = 18, LR = 0,000470, AvgCost = 211,9884, Sent = 636, SentPerMin = 954,36, WordPerSec = 874,40
Update = 16400, Epoch = 18, LR = 0,000469, AvgCost = 211,8138, Sent = 935, SentPerMin = 958,83, WordPerSec = 881,41
Update = 16422, Epoch = 18, LR = 0,000468, AvgCost = 211,4548, Sent = 1000, SentPerMin = 962,23, WordPerSec = 883,10
Update = 16500, Epoch = 19, LR = 0,000467, AvgCost = 215,5916, Sent = 225, SentPerMin = 948,84, WordPerSec = 872,23
Update = 16600, Epoch = 19, LR = 0,000466, AvgCost = 214,0288, Sent = 524, SentPerMin = 950,95, WordPerSec = 875,00
Update = 16700, Epoch = 19, LR = 0,000464, AvgCost = 211,9710, Sent = 819, SentPerMin = 954,43, WordPerSec = 877,88
Update = 16760, Epoch = 19, LR = 0,000463, AvgCost = 210,7617, Sent = 1000, SentPerMin = 960,77, WordPerSec = 881,76
Update = 16800, Epoch = 20, LR = 0,000463, AvgCost = 208,5517, Sent = 122, SentPerMin = 697,52, WordPerSec = 608,80
Update = 16900, Epoch = 20, LR = 0,000462, AvgCost = 214,2613, Sent = 411, SentPerMin = 663,46, WordPerSec = 611,05
Update = 17000, Epoch = 20, LR = 0,000460, AvgCost = 210,2738, Sent = 709, SentPerMin = 676,12, WordPerSec = 617,60
Update = 17098, Epoch = 20, LR = 0,000459, AvgCost = 210,7616, Sent = 1000, SentPerMin = 675,57, WordPerSec = 620,02
Update = 17100, Epoch = 21, LR = 0,000459, AvgCost = 189,9929, Sent = 6, SentPerMin = 833,62, WordPerSec = 757,21
Update = 17200, Epoch = 21, LR = 0,000457, AvgCost = 218,7742, Sent = 292, SentPerMin = 930,49, WordPerSec = 870,58
Update = 17300, Epoch = 21, LR = 0,000456, AvgCost = 211,7450, Sent = 595, SentPerMin = 949,24, WordPerSec = 873,68
Update = 17400, Epoch = 21, LR = 0,000455, AvgCost = 210,8309, Sent = 890, SentPerMin = 954,07, WordPerSec = 879,48
Update = 17436, Epoch = 21, LR = 0,000454, AvgCost = 209,4989, Sent = 1000, SentPerMin = 959,89, WordPerSec = 880,95
Update = 17500, Epoch = 22, LR = 0,000454, AvgCost = 215,9699, Sent = 184, SentPerMin = 935,94, WordPerSec = 859,89
Update = 17600, Epoch = 22, LR = 0,000452, AvgCost = 211,6300, Sent = 482, SentPerMin = 948,01, WordPerSec = 871,57
Update = 17700, Epoch = 22, LR = 0,000451, AvgCost = 211,6912, Sent = 775, SentPerMin = 943,41, WordPerSec = 874,08
Update = 17774, Epoch = 22, LR = 0,000450, AvgCost = 208,5466, Sent = 1000, SentPerMin = 957,54, WordPerSec = 878,80
Update = 17800, Epoch = 23, LR = 0,000450, AvgCost = 203,2324, Sent = 81, SentPerMin = 966,59, WordPerSec = 858,00
Update = 17900, Epoch = 23, LR = 0,000448, AvgCost = 213,0301, Sent = 368, SentPerMin = 938,11, WordPerSec = 868,18
Update = 18000, Epoch = 23, LR = 0,000447, AvgCost = 209,1134, Sent = 663, SentPerMin = 950,32, WordPerSec = 873,14
Update = 18100, Epoch = 23, LR = 0,000446, AvgCost = 206,7716, Sent = 969, SentPerMin = 961,67, WordPerSec = 879,38
Update = 18112, Epoch = 23, LR = 0,000446, AvgCost = 207,7208, Sent = 1000, SentPerMin = 959,10, WordPerSec = 880,23
Update = 18200, Epoch = 24, LR = 0,000445, AvgCost = 216,9631, Sent = 251, SentPerMin = 930,34, WordPerSec = 869,12
Update = 18300, Epoch = 24, LR = 0,000444, AvgCost = 207,9925, Sent = 556, SentPerMin = 952,71, WordPerSec = 872,03
Update = 18400, Epoch = 24, LR = 0,000442, AvgCost = 208,3447, Sent = 847, SentPerMin = 950,30, WordPerSec = 875,89
Update = 18450, Epoch = 24, LR = 0,000442, AvgCost = 206,9880, Sent = 1000, SentPerMin = 957,84, WordPerSec = 879,08
Update = 18500, Epoch = 25, LR = 0,000441, AvgCost = 209,4442, Sent = 145, SentPerMin = 932,83, WordPerSec = 848,45
Update = 18600, Epoch = 25, LR = 0,000440, AvgCost = 212,2788, Sent = 437, SentPerMin = 939,26, WordPerSec = 870,41
Update = 18700, Epoch = 25, LR = 0,000439, AvgCost = 206,3569, Sent = 739, SentPerMin = 951,56, WordPerSec = 872,29
Update = 18788, Epoch = 25, LR = 0,000438, AvgCost = 206,3068, Sent = 1000, SentPerMin = 957,10, WordPerSec = 878,39
Update = 18800, Epoch = 26, LR = 0,000438, AvgCost = 169,0512, Sent = 41, SentPerMin = 1074,91, WordPerSec = 827,59
Update = 18900, Epoch = 26, LR = 0,000436, AvgCost = 212,5185, Sent = 324, SentPerMin = 934,46, WordPerSec = 867,98
Update = 19000, Epoch = 26, LR = 0,000435, AvgCost = 206,7609, Sent = 624, SentPerMin = 949,76, WordPerSec = 870,69
Update = 19100, Epoch = 26, LR = 0,000434, AvgCost = 206,2419, Sent = 921, SentPerMin = 954,58, WordPerSec = 878,26
Update = 19126, Epoch = 26, LR = 0,000434, AvgCost = 205,5249, Sent = 1000, SentPerMin = 958,61, WordPerSec = 879,78
Update = 19200, Epoch = 27, LR = 0,000433, AvgCost = 211,1267, Sent = 213, SentPerMin = 936,78, WordPerSec = 864,21
Update = 19300, Epoch = 27, LR = 0,000432, AvgCost = 208,6832, Sent = 511, SentPerMin = 943,38, WordPerSec = 869,90
Update = 19400, Epoch = 27, LR = 0,000431, AvgCost = 206,8636, Sent = 808, SentPerMin = 953,07, WordPerSec = 877,90
Update = 19464, Epoch = 27, LR = 0,000430, AvgCost = 205,0765, Sent = 1000, SentPerMin = 960,00, WordPerSec = 881,05
Update = 19500, Epoch = 28, LR = 0,000430, AvgCost = 193,7618, Sent = 113, SentPerMin = 998,12, WordPerSec = 855,03
Update = 19600, Epoch = 28, LR = 0,000429, AvgCost = 207,7525, Sent = 399, SentPerMin = 943,99, WordPerSec = 868,91
Update = 19700, Epoch = 28, LR = 0,000427, AvgCost = 204,8966, Sent = 696, SentPerMin = 956,82, WordPerSec = 876,65
Update = 19800, Epoch = 28, LR = 0,000426, AvgCost = 204,3857, Sent = 995, SentPerMin = 961,31, WordPerSec = 881,65
Update = 19802, Epoch = 28, LR = 0,000426, AvgCost = 204,3421, Sent = 1000, SentPerMin = 960,80, WordPerSec = 881,79
Update = 19900, Epoch = 29, LR = 0,000425, AvgCost = 214,1717, Sent = 280, SentPerMin = 922,88, WordPerSec = 865,97
Update = 20000, Epoch = 29, LR = 0,000424, AvgCost = 204,6098, Sent = 588, SentPerMin = 951,35, WordPerSec = 870,24
Update = 20100, Epoch = 29, LR = 0,000423, AvgCost = 205,0894, Sent = 877, SentPerMin = 662,39, WordPerSec = 610,89
Update = 20140, Epoch = 29, LR = 0,000423, AvgCost = 203,8335, Sent = 1000, SentPerMin = 691,75, WordPerSec = 634,87
Update = 20200, Epoch = 30, LR = 0,000422, AvgCost = 209,0742, Sent = 173, SentPerMin = 931,72, WordPerSec = 857,22
Update = 20300, Epoch = 30, LR = 0,000421, AvgCost = 209,9295, Sent = 469, SentPerMin = 906,65, WordPerSec = 836,31
Update = 20400, Epoch = 30, LR = 0,000420, AvgCost = 208,0456, Sent = 766, SentPerMin = 796,40, WordPerSec = 734,87
Update = 20478, Epoch = 30, LR = 0,000419, AvgCost = 205,4200, Sent = 1000, SentPerMin = 771,99, WordPerSec = 708,50
Update = 20500, Epoch = 31, LR = 0,000419, AvgCost = 201,0900, Sent = 71, SentPerMin = 995,95, WordPerSec = 861,99
Update = 20600, Epoch = 31, LR = 0,000418, AvgCost = 209,4128, Sent = 358, SentPerMin = 941,55, WordPerSec = 864,88
Update = 20700, Epoch = 31, LR = 0,000417, AvgCost = 206,1501, Sent = 652, SentPerMin = 947,13, WordPerSec = 869,72
Update = 20800, Epoch = 31, LR = 0,000416, AvgCost = 204,5433, Sent = 955, SentPerMin = 956,06, WordPerSec = 877,12
Update = 20816, Epoch = 31, LR = 0,000416, AvgCost = 204,5285, Sent = 1000, SentPerMin = 956,51, WordPerSec = 877,85
Update = 20900, Epoch = 32, LR = 0,000415, AvgCost = 206,7241, Sent = 244, SentPerMin = 952,25, WordPerSec = 869,97
Update = 21000, Epoch = 32, LR = 0,000414, AvgCost = 206,0763, Sent = 542, SentPerMin = 947,77, WordPerSec = 870,48
Update = 21100, Epoch = 32, LR = 0,000413, AvgCost = 205,0154, Sent = 835, SentPerMin = 949,64, WordPerSec = 873,69
Update = 21154, Epoch = 32, LR = 0,000413, AvgCost = 203,6364, Sent = 1000, SentPerMin = 955,39, WordPerSec = 876,83
Update = 21200, Epoch = 33, LR = 0,000412, AvgCost = 202,1705, Sent = 136, SentPerMin = 959,53, WordPerSec = 853,82
Update = 21300, Epoch = 33, LR = 0,000411, AvgCost = 206,5997, Sent = 428, SentPerMin = 938,42, WordPerSec = 863,14
Update = 21400, Epoch = 33, LR = 0,000410, AvgCost = 203,6046, Sent = 725, SentPerMin = 949,30, WordPerSec = 870,19
Update = 21492, Epoch = 33, LR = 0,000409, AvgCost = 203,1140, Sent = 1000, SentPerMin = 955,74, WordPerSec = 877,14
Update = 21500, Epoch = 34, LR = 0,000409, AvgCost = 188,9817, Sent = 23, SentPerMin = 913,61, WordPerSec = 796,43
Update = 21600, Epoch = 34, LR = 0,000408, AvgCost = 209,7371, Sent = 311, SentPerMin = 932,75, WordPerSec = 867,22
Update = 21700, Epoch = 34, LR = 0,000407, AvgCost = 204,7524, Sent = 611, SentPerMin = 944,00, WordPerSec = 869,45
Update = 21800, Epoch = 34, LR = 0,000406, AvgCost = 203,7900, Sent = 906, SentPerMin = 949,01, WordPerSec = 875,46
Update = 21830, Epoch = 34, LR = 0,000406, AvgCost = 202,2791, Sent = 1000, SentPerMin = 956,16, WordPerSec = 877,54
Update = 21900, Epoch = 35, LR = 0,000405, AvgCost = 207,0883, Sent = 200, SentPerMin = 934,49, WordPerSec = 857,55
Update = 22000, Epoch = 35, LR = 0,000405, AvgCost = 203,8016, Sent = 501, SentPerMin = 947,57, WordPerSec = 868,10
Update = 22100, Epoch = 35, LR = 0,000404, AvgCost = 203,8524, Sent = 795, SentPerMin = 947,74, WordPerSec = 873,26
Update = 22168, Epoch = 35, LR = 0,000403, AvgCost = 201,6565, Sent = 1000, SentPerMin = 957,23, WordPerSec = 878,52
Update = 22200, Epoch = 36, LR = 0,000403, AvgCost = 197,6098, Sent = 99, SentPerMin = 973,98, WordPerSec = 848,70
Update = 22300, Epoch = 36, LR = 0,000402, AvgCost = 204,4093, Sent = 388, SentPerMin = 945,41, WordPerSec = 866,51
Update = 22400, Epoch = 36, LR = 0,000401, AvgCost = 202,2442, Sent = 683, SentPerMin = 953,10, WordPerSec = 873,32
Update = 22500, Epoch = 36, LR = 0,000400, AvgCost = 201,5996, Sent = 984, SentPerMin = 957,32, WordPerSec = 878,53
Update = 22506, Epoch = 36, LR = 0,000400, AvgCost = 201,4691, Sent = 1000, SentPerMin = 956,91, WordPerSec = 878,22
Update = 22600, Epoch = 37, LR = 0,000399, AvgCost = 210,4312, Sent = 269, SentPerMin = 923,74, WordPerSec = 863,41
Update = 22700, Epoch = 37, LR = 0,000398, AvgCost = 203,2095, Sent = 573, SentPerMin = 944,17, WordPerSec = 867,14
Update = 22800, Epoch = 37, LR = 0,000397, AvgCost = 202,2646, Sent = 865, SentPerMin = 945,65, WordPerSec = 870,98
Update = 22844, Epoch = 37, LR = 0,000397, AvgCost = 201,1757, Sent = 1000, SentPerMin = 952,79, WordPerSec = 874,44
Update = 22900, Epoch = 38, LR = 0,000396, AvgCost = 206,2627, Sent = 161, SentPerMin = 930,27, WordPerSec = 853,33
Update = 23000, Epoch = 38, LR = 0,000396, AvgCost = 206,0328, Sent = 456, SentPerMin = 940,22, WordPerSec = 868,74
Update = 23100, Epoch = 38, LR = 0,000395, AvgCost = 202,0661, Sent = 757, SentPerMin = 947,90, WordPerSec = 871,90
Update = 23182, Epoch = 38, LR = 0,000394, AvgCost = 200,7016, Sent = 1000, SentPerMin = 956,42, WordPerSec = 877,77
Update = 23200, Epoch = 39, LR = 0,000394, AvgCost = 193,0685, Sent = 59, SentPerMin = 733,59, WordPerSec = 612,99
Update = 23300, Epoch = 39, LR = 0,000393, AvgCost = 207,3915, Sent = 343, SentPerMin = 666,18, WordPerSec = 615,94
Update = 23400, Epoch = 39, LR = 0,000392, AvgCost = 202,2204, Sent = 640, SentPerMin = 741,58, WordPerSec = 681,46
Update = 23500, Epoch = 39, LR = 0,000391, AvgCost = 200,6872, Sent = 943, SentPerMin = 801,66, WordPerSec = 735,83
Update = 23520, Epoch = 39, LR = 0,000391, AvgCost = 200,6256, Sent = 1000, SentPerMin = 810,09, WordPerSec = 743,48
Update = 23600, Epoch = 40, LR = 0,000391, AvgCost = 204,0317, Sent = 231, SentPerMin = 942,02, WordPerSec = 866,85
Update = 23700, Epoch = 40, LR = 0,000390, AvgCost = 202,5746, Sent = 529, SentPerMin = 877,86, WordPerSec = 807,97
Update = 23800, Epoch = 40, LR = 0,000389, AvgCost = 201,6454, Sent = 824, SentPerMin = 791,14, WordPerSec = 728,00
Update = 23858, Epoch = 40, LR = 0,000388, AvgCost = 200,5499, Sent = 1000, SentPerMin = 772,07, WordPerSec = 708,58
Update = 23900, Epoch = 41, LR = 0,000388, AvgCost = 199,0891, Sent = 126, SentPerMin = 970,06, WordPerSec = 856,38
Update = 24000, Epoch = 41, LR = 0,000387, AvgCost = 201,9755, Sent = 418, SentPerMin = 944,18, WordPerSec = 866,40
Update = 24100, Epoch = 41, LR = 0,000386, AvgCost = 198,9915, Sent = 715, SentPerMin = 956,42, WordPerSec = 874,65
Update = 24196, Epoch = 41, LR = 0,000386, AvgCost = 199,5967, Sent = 1000, SentPerMin = 956,80, WordPerSec = 878,12
Update = 24200, Epoch = 42, LR = 0,000386, AvgCost = 182,6828, Sent = 12, SentPerMin = 969,89, WordPerSec = 872,90
Update = 24300, Epoch = 42, LR = 0,000385, AvgCost = 207,0984, Sent = 299, SentPerMin = 926,79, WordPerSec = 864,49
Update = 24400, Epoch = 42, LR = 0,000384, AvgCost = 200,1621, Sent = 601, SentPerMin = 944,71, WordPerSec = 867,85
Update = 24500, Epoch = 42, LR = 0,000383, AvgCost = 199,5831, Sent = 897, SentPerMin = 948,30, WordPerSec = 872,27
Update = 24534, Epoch = 42, LR = 0,000383, AvgCost = 198,7035, Sent = 1000, SentPerMin = 952,49, WordPerSec = 874,16
Update = 24600, Epoch = 43, LR = 0,000383, AvgCost = 205,1045, Sent = 189, SentPerMin = 933,19, WordPerSec = 859,62
Update = 24700, Epoch = 43, LR = 0,000382, AvgCost = 199,9750, Sent = 490, SentPerMin = 951,23, WordPerSec = 869,60
Update = 24800, Epoch = 43, LR = 0,000381, AvgCost = 200,4943, Sent = 784, SentPerMin = 947,09, WordPerSec = 872,85
Update = 24872, Epoch = 43, LR = 0,000380, AvgCost = 198,2601, Sent = 1000, SentPerMin = 954,89, WordPerSec = 876,37
Update = 24900, Epoch = 44, LR = 0,000380, AvgCost = 193,6975, Sent = 87, SentPerMin = 976,64, WordPerSec = 858,21
Update = 25000, Epoch = 44, LR = 0,000379, AvgCost = 201,6100, Sent = 375, SentPerMin = 936,74, WordPerSec = 863,39
Update = 25100, Epoch = 44, LR = 0,000379, AvgCost = 199,4924, Sent = 668, SentPerMin = 943,88, WordPerSec = 869,77
Update = 25200, Epoch = 44, LR = 0,000378, AvgCost = 196,8034, Sent = 974, SentPerMin = 956,16, WordPerSec = 874,96
Update = 25210, Epoch = 44, LR = 0,000378, AvgCost = 197,7487, Sent = 1000, SentPerMin = 953,85, WordPerSec = 875,41
Update = 25300, Epoch = 45, LR = 0,000377, AvgCost = 205,3838, Sent = 258, SentPerMin = 921,30, WordPerSec = 858,75
Update = 25400, Epoch = 45, LR = 0,000376, AvgCost = 198,0544, Sent = 562, SentPerMin = 947,16, WordPerSec = 868,48
Update = 25500, Epoch = 45, LR = 0,000376, AvgCost = 198,3823, Sent = 854, SentPerMin = 946,71, WordPerSec = 871,81
Update = 25548, Epoch = 45, LR = 0,000375, AvgCost = 197,1802, Sent = 1000, SentPerMin = 953,13, WordPerSec = 874,75
Update = 25600, Epoch = 46, LR = 0,000375, AvgCost = 196,2648, Sent = 152, SentPerMin = 949,07, WordPerSec = 853,86
Update = 25700, Epoch = 46, LR = 0,000374, AvgCost = 201,1951, Sent = 443, SentPerMin = 938,98, WordPerSec = 868,15
Update = 25800, Epoch = 46, LR = 0,000374, AvgCost = 197,1562, Sent = 744, SentPerMin = 946,79, WordPerSec = 869,57
Update = 25886, Epoch = 46, LR = 0,000373, AvgCost = 196,7110, Sent = 1000, SentPerMin = 954,35, WordPerSec = 875,87
Update = 25900, Epoch = 47, LR = 0,000373, AvgCost = 185,4474, Sent = 45, SentPerMin = 1011,90, WordPerSec = 828,63
Update = 26000, Epoch = 47, LR = 0,000372, AvgCost = 202,1869, Sent = 330, SentPerMin = 934,66, WordPerSec = 866,40
Update = 26100, Epoch = 47, LR = 0,000371, AvgCost = 197,3757, Sent = 630, SentPerMin = 946,29, WordPerSec = 867,81
Update = 26200, Epoch = 47, LR = 0,000371, AvgCost = 196,8016, Sent = 929, SentPerMin = 951,53, WordPerSec = 874,21
Update = 26224, Epoch = 47, LR = 0,000371, AvgCost = 196,3237, Sent = 1000, SentPerMin = 955,12, WordPerSec = 876,58
Update = 26300, Epoch = 48, LR = 0,000370, AvgCost = 200,2899, Sent = 219, SentPerMin = 940,50, WordPerSec = 866,64
Update = 26400, Epoch = 48, LR = 0,000369, AvgCost = 198,1948, Sent = 519, SentPerMin = 945,48, WordPerSec = 867,60
Update = 26500, Epoch = 48, LR = 0,000369, AvgCost = 197,3622, Sent = 814, SentPerMin = 949,58, WordPerSec = 873,42
Update = 26562, Epoch = 48, LR = 0,000368, AvgCost = 196,0020, Sent = 1000, SentPerMin = 956,30, WordPerSec = 877,66
Update = 26600, Epoch = 49, LR = 0,000368, AvgCost = 184,5704, Sent = 119, SentPerMin = 716,17, WordPerSec = 609,25
Update = 26700, Epoch = 49, LR = 0,000367, AvgCost = 199,4186, Sent = 406, SentPerMin = 671,93, WordPerSec = 617,56
Update = 26800, Epoch = 49, LR = 0,000367, AvgCost = 196,5933, Sent = 702, SentPerMin = 762,64, WordPerSec = 697,45
Update = 26900, Epoch = 49, LR = 0,000366, AvgCost = 196,4183, Sent = 1000, SentPerMin = 812,48, WordPerSec = 745,67
Update = 26900, Epoch = 49, LR = 0,000366, AvgCost = 196,4183, Sent = 1000, SentPerMin = 812,47, WordPerSec = 745,66
Update = 27000, Epoch = 50, LR = 0,000365, AvgCost = 205,8147, Sent = 285, SentPerMin = 923,28, WordPerSec = 866,97
Update = 27100, Epoch = 50, LR = 0,000364, AvgCost = 196,7759, Sent = 592, SentPerMin = 851,55, WordPerSec = 781,14
Update = 27200, Epoch = 50, LR = 0,000364, AvgCost = 197,9966, Sent = 882, SentPerMin = 781,48, WordPerSec = 721,95
Update = 27238, Epoch = 50, LR = 0,000364, AvgCost = 196,3426, Sent = 1000, SentPerMin = 773,64, WordPerSec = 710,03
Update = 27300, Epoch = 51, LR = 0,000363, AvgCost = 198,2219, Sent = 178, SentPerMin = 928,18, WordPerSec = 856,48
Update = 27400, Epoch = 51, LR = 0,000362, AvgCost = 198,5500, Sent = 475, SentPerMin = 942,35, WordPerSec = 868,02
Update = 27500, Epoch = 51, LR = 0,000362, AvgCost = 197,8583, Sent = 771, SentPerMin = 944,80, WordPerSec = 873,28
Update = 27576, Epoch = 51, LR = 0,000361, AvgCost = 195,4665, Sent = 1000, SentPerMin = 956,18, WordPerSec = 877,55
Update = 27600, Epoch = 52, LR = 0,000361, AvgCost = 190,8710, Sent = 76, SentPerMin = 976,05, WordPerSec = 860,89
Update = 27700, Epoch = 52, LR = 0,000361, AvgCost = 198,9407, Sent = 363, SentPerMin = 938,59, WordPerSec = 865,50
Update = 27800, Epoch = 52, LR = 0,000360, AvgCost = 196,3371, Sent = 656, SentPerMin = 945,14, WordPerSec = 870,22
Update = 27900, Epoch = 52, LR = 0,000359, AvgCost = 194,4104, Sent = 961, SentPerMin = 955,01, WordPerSec = 875,33
Update = 27914, Epoch = 52, LR = 0,000359, AvgCost = 194,7339, Sent = 1000, SentPerMin = 954,58, WordPerSec = 876,08
Update = 28000, Epoch = 53, LR = 0,000359, AvgCost = 199,1680, Sent = 248, SentPerMin = 943,82, WordPerSec = 871,89
Update = 28100, Epoch = 53, LR = 0,000358, AvgCost = 195,4608, Sent = 549, SentPerMin = 950,54, WordPerSec = 870,92
Update = 28200, Epoch = 53, LR = 0,000357, AvgCost = 195,6267, Sent = 841, SentPerMin = 948,85, WordPerSec = 873,75
Update = 28252, Epoch = 53, LR = 0,000357, AvgCost = 194,2794, Sent = 1000, SentPerMin = 955,17, WordPerSec = 876,62
Update = 28300, Epoch = 54, LR = 0,000357, AvgCost = 192,9447, Sent = 140, SentPerMin = 952,00, WordPerSec = 854,53
Update = 28400, Epoch = 54, LR = 0,000356, AvgCost = 199,2889, Sent = 431, SentPerMin = 938,13, WordPerSec = 869,60
Update = 28500, Epoch = 54, LR = 0,000355, AvgCost = 193,5975, Sent = 732, SentPerMin = 955,63, WordPerSec = 873,68
Update = 28590, Epoch = 54, LR = 0,000355, AvgCost = 193,8379, Sent = 1000, SentPerMin = 957,41, WordPerSec = 878,68
Update = 28600, Epoch = 55, LR = 0,000355, AvgCost = 160,4890, Sent = 33, SentPerMin = 1006,49, WordPerSec = 777,75
Update = 28700, Epoch = 55, LR = 0,000354, AvgCost = 200,0998, Sent = 317, SentPerMin = 929,85, WordPerSec = 865,91
Update = 28800, Epoch = 55, LR = 0,000354, AvgCost = 194,6708, Sent = 618, SentPerMin = 946,51, WordPerSec = 868,68
Update = 28900, Epoch = 55, LR = 0,000353, AvgCost = 194,4329, Sent = 914, SentPerMin = 949,33, WordPerSec = 874,81
Update = 28928, Epoch = 55, LR = 0,000353, AvgCost = 193,3099, Sent = 1000, SentPerMin = 954,82, WordPerSec = 876,31
Update = 29000, Epoch = 56, LR = 0,000352, AvgCost = 200,3398, Sent = 205, SentPerMin = 926,03, WordPerSec = 864,60
Update = 29100, Epoch = 56, LR = 0,000352, AvgCost = 194,5382, Sent = 506, SentPerMin = 944,50, WordPerSec = 866,32
Update = 29200, Epoch = 56, LR = 0,000351, AvgCost = 194,7788, Sent = 802, SentPerMin = 949,74, WordPerSec = 874,37
Update = 29266, Epoch = 56, LR = 0,000351, AvgCost = 192,9099, Sent = 1000, SentPerMin = 956,90, WordPerSec = 878,21
Update = 29300, Epoch = 57, LR = 0,000351, AvgCost = 183,8731, Sent = 106, SentPerMin = 990,37, WordPerSec = 853,03
Update = 29400, Epoch = 57, LR = 0,000350, AvgCost = 195,3561, Sent = 393, SentPerMin = 932,18, WordPerSec = 859,13
Update = 29500, Epoch = 57, LR = 0,000349, AvgCost = 192,7593, Sent = 689, SentPerMin = 948,29, WordPerSec = 869,27
Update = 29600, Epoch = 57, LR = 0,000349, AvgCost = 192,4608, Sent = 989, SentPerMin = 949,53, WordPerSec = 871,69
Update = 29604, Epoch = 57, LR = 0,000349, AvgCost = 192,2995, Sent = 1000, SentPerMin = 949,62, WordPerSec = 871,53
Update = 29700, Epoch = 58, LR = 0,000348, AvgCost = 201,8451, Sent = 275, SentPerMin = 927,29, WordPerSec = 870,47
Update = 29800, Epoch = 58, LR = 0,000348, AvgCost = 193,1656, Sent = 581, SentPerMin = 949,04, WordPerSec = 870,60
Update = 29900, Epoch = 58, LR = 0,000347, AvgCost = 193,5167, Sent = 870, SentPerMin = 947,78, WordPerSec = 875,06
Update = 29942, Epoch = 58, LR = 0,000347, AvgCost = 192,0860, Sent = 1000, SentPerMin = 955,14, WordPerSec = 876,60
Update = 30000, Epoch = 59, LR = 0,000346, AvgCost = 196,4779, Sent = 166, SentPerMin = 666,37, WordPerSec = 614,05
Update = 30100, Epoch = 59, LR = 0,000346, AvgCost = 197,6775, Sent = 463, SentPerMin = 483,26, WordPerSec = 446,05
Update = 30200, Epoch = 59, LR = 0,000345, AvgCost = 196,0014, Sent = 761, SentPerMin = 545,60, WordPerSec = 502,41
Update = 30280, Epoch = 59, LR = 0,000345, AvgCost = 193,9455, Sent = 1000, SentPerMin = 575,05, WordPerSec = 527,76
Update = 30300, Epoch = 60, LR = 0,000345, AvgCost = 188,8569, Sent = 64, SentPerMin = 995,77, WordPerSec = 868,19
Update = 30400, Epoch = 60, LR = 0,000344, AvgCost = 196,6084, Sent = 352, SentPerMin = 942,91, WordPerSec = 867,10
Update = 30500, Epoch = 60, LR = 0,000344, AvgCost = 194,1283, Sent = 645, SentPerMin = 946,96, WordPerSec = 871,64
Update = 30600, Epoch = 60, LR = 0,000343, AvgCost = 193,0047, Sent = 948, SentPerMin = 954,36, WordPerSec = 877,35
Update = 30618, Epoch = 60, LR = 0,000343, AvgCost = 192,5322, Sent = 1000, SentPerMin = 956,18, WordPerSec = 877,55
Update = 30700, Epoch = 61, LR = 0,000342, AvgCost = 193,7754, Sent = 239, SentPerMin = 946,20, WordPerSec = 864,12
Update = 30800, Epoch = 61, LR = 0,000342, AvgCost = 194,1149, Sent = 536, SentPerMin = 943,18, WordPerSec = 866,96
Update = 30900, Epoch = 61, LR = 0,000341, AvgCost = 192,7077, Sent = 831, SentPerMin = 948,46, WordPerSec = 870,91
Update = 30956, Epoch = 61, LR = 0,000341, AvgCost = 191,8242, Sent = 1000, SentPerMin = 954,37, WordPerSec = 875,89
Update = 31000, Epoch = 62, LR = 0,000341, AvgCost = 189,8551, Sent = 131, SentPerMin = 960,84, WordPerSec = 855,10
Update = 31100, Epoch = 62, LR = 0,000340, AvgCost = 193,7612, Sent = 423, SentPerMin = 940,93, WordPerSec = 864,71
Update = 31200, Epoch = 62, LR = 0,000340, AvgCost = 191,4359, Sent = 720, SentPerMin = 953,37, WordPerSec = 872,49
Update = 31294, Epoch = 62, LR = 0,000339, AvgCost = 191,3349, Sent = 1000, SentPerMin = 953,82, WordPerSec = 875,39
Update = 31300, Epoch = 63, LR = 0,000339, AvgCost = 191,2423, Sent = 16, SentPerMin = 871,76, WordPerSec = 850,87
Update = 31400, Epoch = 63, LR = 0,000339, AvgCost = 198,4451, Sent = 304, SentPerMin = 929,67, WordPerSec = 868,31
Update = 31500, Epoch = 63, LR = 0,000338, AvgCost = 192,5995, Sent = 606, SentPerMin = 945,94, WordPerSec = 869,37
Update = 31600, Epoch = 63, LR = 0,000338, AvgCost = 192,1907, Sent = 902, SentPerMin = 950,08, WordPerSec = 875,01
Update = 31632, Epoch = 63, LR = 0,000337, AvgCost = 190,9755, Sent = 1000, SentPerMin = 955,45, WordPerSec = 876,88
Update = 31700, Epoch = 64, LR = 0,000337, AvgCost = 195,5646, Sent = 195, SentPerMin = 927,44, WordPerSec = 852,45
Update = 31800, Epoch = 64, LR = 0,000336, AvgCost = 192,2192, Sent = 495, SentPerMin = 947,80, WordPerSec = 866,20
Update = 31900, Epoch = 64, LR = 0,000336, AvgCost = 193,0237, Sent = 789, SentPerMin = 943,96, WordPerSec = 870,40
Update = 31970, Epoch = 64, LR = 0,000336, AvgCost = 190,6480, Sent = 1000, SentPerMin = 954,67, WordPerSec = 876,17
Update = 32000, Epoch = 65, LR = 0,000335, AvgCost = 187,9615, Sent = 92, SentPerMin = 957,78, WordPerSec = 850,03
Update = 32100, Epoch = 65, LR = 0,000335, AvgCost = 194,0748, Sent = 381, SentPerMin = 937,03, WordPerSec = 862,27
Update = 32200, Epoch = 65, LR = 0,000334, AvgCost = 191,8356, Sent = 676, SentPerMin = 947,03, WordPerSec = 869,44
Update = 32300, Epoch = 65, LR = 0,000334, AvgCost = 189,4878, Sent = 979, SentPerMin = 955,23, WordPerSec = 874,67
Update = 32308, Epoch = 65, LR = 0,000334, AvgCost = 190,4429, Sent = 1000, SentPerMin = 953,23, WordPerSec = 874,85
Update = 32400, Epoch = 66, LR = 0,000333, AvgCost = 196,8712, Sent = 265, SentPerMin = 931,35, WordPerSec = 864,34
Update = 32500, Epoch = 66, LR = 0,000333, AvgCost = 192,1108, Sent = 566, SentPerMin = 944,60, WordPerSec = 869,39
Update = 32600, Epoch = 66, LR = 0,000332, AvgCost = 191,3823, Sent = 859, SentPerMin = 945,96, WordPerSec = 871,70
Update = 32646, Epoch = 66, LR = 0,000332, AvgCost = 189,9831, Sent = 1000, SentPerMin = 954,14, WordPerSec = 875,68
Update = 32700, Epoch = 67, LR = 0,000332, AvgCost = 191,8473, Sent = 157, SentPerMin = 938,75, WordPerSec = 853,94
Update = 32800, Epoch = 67, LR = 0,000331, AvgCost = 194,2764, Sent = 450, SentPerMin = 936,46, WordPerSec = 863,52
Update = 32900, Epoch = 67, LR = 0,000331, AvgCost = 190,2457, Sent = 752, SentPerMin = 946,84, WordPerSec = 868,09
Update = 32984, Epoch = 67, LR = 0,000330, AvgCost = 189,7971, Sent = 1000, SentPerMin = 954,08, WordPerSec = 875,62
Update = 33000, Epoch = 68, LR = 0,000330, AvgCost = 187,0684, Sent = 51, SentPerMin = 711,44, WordPerSec = 613,33
Update = 33100, Epoch = 68, LR = 0,000330, AvgCost = 198,1038, Sent = 336, SentPerMin = 662,87, WordPerSec = 614,14
Update = 33200, Epoch = 68, LR = 0,000329, AvgCost = 191,9089, Sent = 636, SentPerMin = 740,82, WordPerSec = 678,75
Update = 33300, Epoch = 68, LR = 0,000329, AvgCost = 191,0686, Sent = 935, SentPerMin = 797,35, WordPerSec = 732,96
Update = 33322, Epoch = 68, LR = 0,000329, AvgCost = 190,5382, Sent = 1000, SentPerMin = 808,53, WordPerSec = 742,04
Update = 33400, Epoch = 69, LR = 0,000328, AvgCost = 194,0391, Sent = 225, SentPerMin = 940,56, WordPerSec = 864,62
Update = 33500, Epoch = 69, LR = 0,000328, AvgCost = 192,8552, Sent = 524, SentPerMin = 884,88, WordPerSec = 814,20
Update = 33600, Epoch = 69, LR = 0,000327, AvgCost = 191,8030, Sent = 819, SentPerMin = 796,88, WordPerSec = 732,97
Update = 33660, Epoch = 69, LR = 0,000327, AvgCost = 190,6398, Sent = 1000, SentPerMin = 774,45, WordPerSec = 710,76
Update = 33700, Epoch = 70, LR = 0,000327, AvgCost = 187,4531, Sent = 122, SentPerMin = 972,33, WordPerSec = 848,66
Update = 33800, Epoch = 70, LR = 0,000326, AvgCost = 192,9031, Sent = 411, SentPerMin = 939,83, WordPerSec = 865,59
Update = 33900, Epoch = 70, LR = 0,000326, AvgCost = 189,4783, Sent = 709, SentPerMin = 956,87, WordPerSec = 874,05
Update = 33998, Epoch = 70, LR = 0,000325, AvgCost = 189,6521, Sent = 1000, SentPerMin = 955,63, WordPerSec = 877,05
Update = 34000, Epoch = 71, LR = 0,000325, AvgCost = 167,0572, Sent = 6, SentPerMin = 906,21, WordPerSec = 823,14
Update = 34100, Epoch = 71, LR = 0,000325, AvgCost = 198,4228, Sent = 292, SentPerMin = 923,53, WordPerSec = 864,07
Update = 34200, Epoch = 71, LR = 0,000324, AvgCost = 191,4552, Sent = 595, SentPerMin = 943,91, WordPerSec = 868,77
Update = 34300, Epoch = 71, LR = 0,000324, AvgCost = 190,7291, Sent = 890, SentPerMin = 948,67, WordPerSec = 874,50
Update = 34336, Epoch = 71, LR = 0,000324, AvgCost = 189,3481, Sent = 1000, SentPerMin = 954,44, WordPerSec = 875,95
Update = 34400, Epoch = 72, LR = 0,000323, AvgCost = 195,2488, Sent = 184, SentPerMin = 930,07, WordPerSec = 854,50
Update = 34500, Epoch = 72, LR = 0,000323, AvgCost = 191,6773, Sent = 482, SentPerMin = 941,45, WordPerSec = 865,53
Update = 34600, Epoch = 72, LR = 0,000323, AvgCost = 191,8290, Sent = 775, SentPerMin = 937,99, WordPerSec = 869,06
Update = 34674, Epoch = 72, LR = 0,000322, AvgCost = 188,8475, Sent = 1000, SentPerMin = 952,46, WordPerSec = 874,13
Starting inference...
Inference results:


在
了 了了了了了了。
在  在在了
*/

            /*
             Update = 10100, Epoch = 0, LR = 0,000597, AvgCost = 266,6838, Sent = 285, SentPerMin = 635,53, WordPerSec = 596,77
Update = 10200, Epoch = 0, LR = 0,000594, AvgCost = 248,0586, Sent = 592, SentPerMin = 673,28, WordPerSec = 617,61
Update = 10300, Epoch = 0, LR = 0,000591, AvgCost = 245,4945, Sent = 882, SentPerMin = 675,54, WordPerSec = 624,09
Update = 10338, Epoch = 0, LR = 0,000590, AvgCost = 242,3422, Sent = 1000, SentPerMin = 682,72, WordPerSec = 626,58
Update = 10400, Epoch = 1, LR = 0,000588, AvgCost = 232,7955, Sent = 178, SentPerMin = 964,70, WordPerSec = 890,18
Update = 10500, Epoch = 1, LR = 0,000586, AvgCost = 233,8184, Sent = 475, SentPerMin = 980,23, WordPerSec = 902,91
Update = 10600, Epoch = 1, LR = 0,000583, AvgCost = 232,4101, Sent = 771, SentPerMin = 978,43, WordPerSec = 904,36
Update = 10676, Epoch = 1, LR = 0,000581, AvgCost = 229,2611, Sent = 1000, SentPerMin = 991,54, WordPerSec = 910,00
Update = 10700, Epoch = 2, LR = 0,000580, AvgCost = 220,9584, Sent = 76, SentPerMin = 1014,38, WordPerSec = 894,70
Update = 10800, Epoch = 2, LR = 0,000577, AvgCost = 231,0768, Sent = 363, SentPerMin = 971,98, WordPerSec = 896,29
Update = 10900, Epoch = 2, LR = 0,000575, AvgCost = 227,7855, Sent = 656, SentPerMin = 979,20, WordPerSec = 901,58
Update = 11000, Epoch = 2, LR = 0,000572, AvgCost = 225,3552, Sent = 961, SentPerMin = 991,91, WordPerSec = 909,15
Update = 11014, Epoch = 2, LR = 0,000572, AvgCost = 225,7901, Sent = 1000, SentPerMin = 991,02, WordPerSec = 909,52
Update = 11100, Epoch = 3, LR = 0,000569, AvgCost = 228,8307, Sent = 248, SentPerMin = 979,97, WordPerSec = 905,28
Update = 11200, Epoch = 3, LR = 0,000567, AvgCost = 224,7438, Sent = 549, SentPerMin = 986,28, WordPerSec = 903,67
Update = 11300, Epoch = 3, LR = 0,000564, AvgCost = 224,7997, Sent = 841, SentPerMin = 985,61, WordPerSec = 907,60
Update = 11352, Epoch = 3, LR = 0,000563, AvgCost = 223,4438, Sent = 1000, SentPerMin = 992,60, WordPerSec = 910,98
Update = 11400, Epoch = 4, LR = 0,000562, AvgCost = 222,2286, Sent = 140, SentPerMin = 975,75, WordPerSec = 875,85
Update = 11500, Epoch = 4, LR = 0,000560, AvgCost = 228,4519, Sent = 431, SentPerMin = 967,04, WordPerSec = 896,40
Update = 11600, Epoch = 4, LR = 0,000557, AvgCost = 221,4071, Sent = 732, SentPerMin = 986,98, WordPerSec = 902,35
Update = 11690, Epoch = 4, LR = 0,000555, AvgCost = 221,7462, Sent = 1000, SentPerMin = 989,81, WordPerSec = 908,42
Update = 11700, Epoch = 5, LR = 0,000555, AvgCost = 183,2840, Sent = 33, SentPerMin = 1080,03, WordPerSec = 834,57
Update = 11800, Epoch = 5, LR = 0,000552, AvgCost = 227,9567, Sent = 317, SentPerMin = 967,60, WordPerSec = 901,06
Update = 11900, Epoch = 5, LR = 0,000550, AvgCost = 221,9655, Sent = 618, SentPerMin = 984,87, WordPerSec = 903,89
Update = 12000, Epoch = 5, LR = 0,000548, AvgCost = 221,5140, Sent = 914, SentPerMin = 987,59, WordPerSec = 910,06
Update = 12028, Epoch = 5, LR = 0,000547, AvgCost = 220,3892, Sent = 1000, SentPerMin = 992,35, WordPerSec = 910,74
Update = 12100, Epoch = 6, LR = 0,000545, AvgCost = 228,7724, Sent = 205, SentPerMin = 961,60, WordPerSec = 897,80
Update = 12200, Epoch = 6, LR = 0,000543, AvgCost = 221,5214, Sent = 506, SentPerMin = 979,19, WordPerSec = 898,14
Update = 12300, Epoch = 6, LR = 0,000541, AvgCost = 221,4306, Sent = 802, SentPerMin = 975,02, WordPerSec = 897,64
Update = 12366, Epoch = 6, LR = 0,000540, AvgCost = 219,5629, Sent = 1000, SentPerMin = 977,75, WordPerSec = 897,34
Update = 12400, Epoch = 7, LR = 0,000539, AvgCost = 209,0092, Sent = 106, SentPerMin = 1017,59, WordPerSec = 876,47
Update = 12500, Epoch = 7, LR = 0,000537, AvgCost = 222,5724, Sent = 393, SentPerMin = 965,02, WordPerSec = 889,39
Update = 12600, Epoch = 7, LR = 0,000535, AvgCost = 219,1844, Sent = 689, SentPerMin = 981,80, WordPerSec = 899,98
Update = 12700, Epoch = 7, LR = 0,000532, AvgCost = 218,9031, Sent = 989, SentPerMin = 986,55, WordPerSec = 905,66
Update = 12704, Epoch = 7, LR = 0,000532, AvgCost = 218,7480, Sent = 1000, SentPerMin = 986,59, WordPerSec = 905,46
Update = 12800, Epoch = 8, LR = 0,000530, AvgCost = 228,6411, Sent = 275, SentPerMin = 957,35, WordPerSec = 898,69
Update = 12900, Epoch = 8, LR = 0,000528, AvgCost = 219,1010, Sent = 581, SentPerMin = 980,23, WordPerSec = 899,22
Update = 13000, Epoch = 8, LR = 0,000526, AvgCost = 219,5209, Sent = 870, SentPerMin = 980,48, WordPerSec = 905,25
Update = 13042, Epoch = 8, LR = 0,000525, AvgCost = 218,1207, Sent = 1000, SentPerMin = 988,64, WordPerSec = 907,34
Update = 13100, Epoch = 9, LR = 0,000524, AvgCost = 220,4048, Sent = 166, SentPerMin = 941,86, WordPerSec = 867,91
Update = 13200, Epoch = 9, LR = 0,000522, AvgCost = 221,5465, Sent = 463, SentPerMin = 965,15, WordPerSec = 890,84
Update = 13300, Epoch = 9, LR = 0,000520, AvgCost = 219,7861, Sent = 761, SentPerMin = 972,33, WordPerSec = 895,37
Update = 13380, Epoch = 9, LR = 0,000519, AvgCost = 217,8251, Sent = 1000, SentPerMin = 983,57, WordPerSec = 902,69
Update = 13400, Epoch = 10, LR = 0,000518, AvgCost = 214,6268, Sent = 64, SentPerMin = 739,74, WordPerSec = 644,96
Update = 13500, Epoch = 10, LR = 0,000516, AvgCost = 221,3254, Sent = 352, SentPerMin = 694,19, WordPerSec = 638,38
Update = 13600, Epoch = 10, LR = 0,000514, AvgCost = 218,7343, Sent = 645, SentPerMin = 694,29, WordPerSec = 639,07
Update = 13700, Epoch = 10, LR = 0,000513, AvgCost = 217,9269, Sent = 948, SentPerMin = 698,39, WordPerSec = 642,03
Update = 13718, Epoch = 10, LR = 0,000512, AvgCost = 217,4127, Sent = 1000, SentPerMin = 698,55, WordPerSec = 641,11
Update = 13800, Epoch = 11, LR = 0,000511, AvgCost = 219,0814, Sent = 239, SentPerMin = 992,73, WordPerSec = 906,61
Update = 13900, Epoch = 11, LR = 0,000509, AvgCost = 218,8484, Sent = 536, SentPerMin = 983,92, WordPerSec = 904,40
Update = 14000, Epoch = 11, LR = 0,000507, AvgCost = 217,5709, Sent = 831, SentPerMin = 987,20, WordPerSec = 906,48
Update = 14056, Epoch = 11, LR = 0,000506, AvgCost = 216,8407, Sent = 1000, SentPerMin = 992,69, WordPerSec = 911,06
Update = 14100, Epoch = 12, LR = 0,000505, AvgCost = 215,0754, Sent = 131, SentPerMin = 992,84, WordPerSec = 883,58
Update = 14200, Epoch = 12, LR = 0,000504, AvgCost = 218,1230, Sent = 423, SentPerMin = 980,81, WordPerSec = 901,35
Update = 14300, Epoch = 12, LR = 0,000502, AvgCost = 215,7392, Sent = 720, SentPerMin = 993,15, WordPerSec = 908,90
Update = 14394, Epoch = 12, LR = 0,000500, AvgCost = 216,0207, Sent = 1000, SentPerMin = 994,52, WordPerSec = 912,73
Update = 14400, Epoch = 13, LR = 0,000500, AvgCost = 217,1209, Sent = 16, SentPerMin = 873,69, WordPerSec = 852,76
Update = 14500, Epoch = 13, LR = 0,000498, AvgCost = 223,1525, Sent = 304, SentPerMin = 963,87, WordPerSec = 900,25
Update = 14600, Epoch = 13, LR = 0,000497, AvgCost = 216,6012, Sent = 606, SentPerMin = 979,25, WordPerSec = 899,99
Update = 14700, Epoch = 13, LR = 0,000495, AvgCost = 216,3830, Sent = 902, SentPerMin = 986,50, WordPerSec = 908,56
Update = 14732, Epoch = 13, LR = 0,000494, AvgCost = 215,1443, Sent = 1000, SentPerMin = 991,85, WordPerSec = 910,29
Update = 14800, Epoch = 14, LR = 0,000493, AvgCost = 220,1517, Sent = 195, SentPerMin = 969,85, WordPerSec = 891,43
Update = 14900, Epoch = 14, LR = 0,000492, AvgCost = 215,6054, Sent = 495, SentPerMin = 985,06, WordPerSec = 900,25
Update = 15000, Epoch = 14, LR = 0,000490, AvgCost = 217,0124, Sent = 789, SentPerMin = 980,93, WordPerSec = 904,49
Update = 15070, Epoch = 14, LR = 0,000489, AvgCost = 214,5782, Sent = 1000, SentPerMin = 991,53, WordPerSec = 909,99
Update = 15100, Epoch = 15, LR = 0,000488, AvgCost = 211,6364, Sent = 92, SentPerMin = 1001,38, WordPerSec = 888,73
Update = 15200, Epoch = 15, LR = 0,000487, AvgCost = 217,0928, Sent = 381, SentPerMin = 975,01, WordPerSec = 897,21
Update = 15300, Epoch = 15, LR = 0,000485, AvgCost = 214,9326, Sent = 676, SentPerMin = 985,54, WordPerSec = 904,80
Update = 15400, Epoch = 15, LR = 0,000483, AvgCost = 212,6881, Sent = 979, SentPerMin = 993,07, WordPerSec = 909,32
Update = 15408, Epoch = 15, LR = 0,000483, AvgCost = 213,7573, Sent = 1000, SentPerMin = 990,38, WordPerSec = 908,94
Update = 15500, Epoch = 16, LR = 0,000482, AvgCost = 220,0637, Sent = 265, SentPerMin = 966,62, WordPerSec = 897,07
Update = 15600, Epoch = 16, LR = 0,000480, AvgCost = 214,6571, Sent = 566, SentPerMin = 970,87, WordPerSec = 893,56
Update = 15700, Epoch = 16, LR = 0,000479, AvgCost = 214,1060, Sent = 859, SentPerMin = 975,71, WordPerSec = 899,12
Update = 15746, Epoch = 16, LR = 0,000478, AvgCost = 212,7902, Sent = 1000, SentPerMin = 984,92, WordPerSec = 903,92
Update = 15800, Epoch = 17, LR = 0,000477, AvgCost = 215,0894, Sent = 157, SentPerMin = 967,69, WordPerSec = 880,27
Update = 15900, Epoch = 17, LR = 0,000476, AvgCost = 217,1961, Sent = 450, SentPerMin = 960,33, WordPerSec = 885,53
Update = 16000, Epoch = 17, LR = 0,000474, AvgCost = 212,8707, Sent = 752, SentPerMin = 976,66, WordPerSec = 895,43
Update = 16084, Epoch = 17, LR = 0,000473, AvgCost = 212,6367, Sent = 1000, SentPerMin = 983,44, WordPerSec = 902,57
Update = 16100, Epoch = 18, LR = 0,000473, AvgCost = 205,0056, Sent = 51, SentPerMin = 1037,56, WordPerSec = 894,47
Update = 16200, Epoch = 18, LR = 0,000471, AvgCost = 217,8380, Sent = 336, SentPerMin = 969,11, WordPerSec = 897,87
Update = 16300, Epoch = 18, LR = 0,000470, AvgCost = 212,3866, Sent = 636, SentPerMin = 982,81, WordPerSec = 900,47
Update = 16400, Epoch = 18, LR = 0,000469, AvgCost = 212,1745, Sent = 935, SentPerMin = 986,44, WordPerSec = 906,79
Update = 16422, Epoch = 18, LR = 0,000468, AvgCost = 211,7021, Sent = 1000, SentPerMin = 990,19, WordPerSec = 908,76
Update = 16500, Epoch = 19, LR = 0,000467, AvgCost = 214,3556, Sent = 225, SentPerMin = 970,51, WordPerSec = 892,15
Update = 16600, Epoch = 19, LR = 0,000466, AvgCost = 213,8917, Sent = 524, SentPerMin = 976,24, WordPerSec = 898,27
Update = 16700, Epoch = 19, LR = 0,000464, AvgCost = 212,5055, Sent = 819, SentPerMin = 982,12, WordPerSec = 903,36
Update = 16760, Epoch = 19, LR = 0,000463, AvgCost = 211,1937, Sent = 1000, SentPerMin = 988,97, WordPerSec = 907,64
Update = 16800, Epoch = 20, LR = 0,000463, AvgCost = 207,3861, Sent = 122, SentPerMin = 725,64, WordPerSec = 633,35
Update = 16900, Epoch = 20, LR = 0,000462, AvgCost = 214,0315, Sent = 411, SentPerMin = 689,46, WordPerSec = 635,00
Update = 17000, Epoch = 20, LR = 0,000460, AvgCost = 210,8459, Sent = 709, SentPerMin = 699,11, WordPerSec = 638,60
Update = 17098, Epoch = 20, LR = 0,000459, AvgCost = 211,2418, Sent = 1000, SentPerMin = 696,66, WordPerSec = 639,37
Update = 17100, Epoch = 21, LR = 0,000459, AvgCost = 188,3839, Sent = 6, SentPerMin = 956,47, WordPerSec = 868,79
Update = 17200, Epoch = 21, LR = 0,000457, AvgCost = 218,8294, Sent = 292, SentPerMin = 963,21, WordPerSec = 901,20
Update = 17300, Epoch = 21, LR = 0,000456, AvgCost = 212,0855, Sent = 595, SentPerMin = 979,29, WordPerSec = 901,33
Update = 17400, Epoch = 21, LR = 0,000455, AvgCost = 211,5473, Sent = 890, SentPerMin = 984,64, WordPerSec = 907,66
Update = 17436, Epoch = 21, LR = 0,000454, AvgCost = 210,1758, Sent = 1000, SentPerMin = 991,49, WordPerSec = 909,96
Update = 17500, Epoch = 22, LR = 0,000454, AvgCost = 215,7206, Sent = 184, SentPerMin = 961,08, WordPerSec = 882,99
Update = 17600, Epoch = 22, LR = 0,000452, AvgCost = 211,9349, Sent = 482, SentPerMin = 974,74, WordPerSec = 896,14
Update = 17700, Epoch = 22, LR = 0,000451, AvgCost = 212,5927, Sent = 775, SentPerMin = 968,65, WordPerSec = 897,47
Update = 17774, Epoch = 22, LR = 0,000450, AvgCost = 209,4129, Sent = 1000, SentPerMin = 982,39, WordPerSec = 901,60
Update = 17800, Epoch = 23, LR = 0,000450, AvgCost = 203,3938, Sent = 81, SentPerMin = 994,93, WordPerSec = 883,15
Update = 17900, Epoch = 23, LR = 0,000448, AvgCost = 213,2389, Sent = 368, SentPerMin = 965,66, WordPerSec = 893,67
Update = 18000, Epoch = 23, LR = 0,000447, AvgCost = 209,6634, Sent = 663, SentPerMin = 978,28, WordPerSec = 898,82
Update = 18100, Epoch = 23, LR = 0,000446, AvgCost = 207,5399, Sent = 969, SentPerMin = 991,25, WordPerSec = 906,43
Update = 18112, Epoch = 23, LR = 0,000446, AvgCost = 208,4520, Sent = 1000, SentPerMin = 988,54, WordPerSec = 907,24
Update = 18200, Epoch = 24, LR = 0,000445, AvgCost = 216,8256, Sent = 251, SentPerMin = 963,71, WordPerSec = 900,29
Update = 18300, Epoch = 24, LR = 0,000444, AvgCost = 208,4022, Sent = 556, SentPerMin = 987,97, WordPerSec = 904,30
Update = 18400, Epoch = 24, LR = 0,000442, AvgCost = 208,9480, Sent = 847, SentPerMin = 983,60, WordPerSec = 906,59
Update = 18450, Epoch = 24, LR = 0,000442, AvgCost = 207,7309, Sent = 1000, SentPerMin = 992,00, WordPerSec = 910,43
Update = 18500, Epoch = 25, LR = 0,000441, AvgCost = 209,0179, Sent = 145, SentPerMin = 972,75, WordPerSec = 884,76
Update = 18600, Epoch = 25, LR = 0,000440, AvgCost = 212,2228, Sent = 437, SentPerMin = 969,01, WordPerSec = 897,97
Update = 18700, Epoch = 25, LR = 0,000439, AvgCost = 206,9877, Sent = 739, SentPerMin = 983,62, WordPerSec = 901,67
Update = 18788, Epoch = 25, LR = 0,000438, AvgCost = 207,0787, Sent = 1000, SentPerMin = 989,89, WordPerSec = 908,48
Update = 18800, Epoch = 26, LR = 0,000438, AvgCost = 169,5705, Sent = 41, SentPerMin = 1099,26, WordPerSec = 846,34
Update = 18900, Epoch = 26, LR = 0,000436, AvgCost = 212,2152, Sent = 324, SentPerMin = 962,66, WordPerSec = 894,17
Update = 19000, Epoch = 26, LR = 0,000435, AvgCost = 207,3210, Sent = 624, SentPerMin = 980,30, WordPerSec = 898,69
Update = 19100, Epoch = 26, LR = 0,000434, AvgCost = 206,9945, Sent = 921, SentPerMin = 986,20, WordPerSec = 907,35
Update = 19126, Epoch = 26, LR = 0,000434, AvgCost = 206,2925, Sent = 1000, SentPerMin = 990,01, WordPerSec = 908,60
Update = 19200, Epoch = 27, LR = 0,000433, AvgCost = 210,5276, Sent = 213, SentPerMin = 969,39, WordPerSec = 894,30
Update = 19300, Epoch = 27, LR = 0,000432, AvgCost = 208,8791, Sent = 511, SentPerMin = 974,49, WordPerSec = 898,59
Update = 19400, Epoch = 27, LR = 0,000431, AvgCost = 207,5724, Sent = 808, SentPerMin = 982,55, WordPerSec = 905,04
Update = 19464, Epoch = 27, LR = 0,000430, AvgCost = 205,8727, Sent = 1000, SentPerMin = 988,36, WordPerSec = 907,08
Update = 19500, Epoch = 28, LR = 0,000430, AvgCost = 193,6553, Sent = 113, SentPerMin = 1022,52, WordPerSec = 875,93
Update = 19600, Epoch = 28, LR = 0,000429, AvgCost = 207,6276, Sent = 399, SentPerMin = 968,80, WordPerSec = 891,75
Update = 19700, Epoch = 28, LR = 0,000427, AvgCost = 205,7265, Sent = 696, SentPerMin = 985,26, WordPerSec = 902,71
Update = 19800, Epoch = 28, LR = 0,000426, AvgCost = 205,4060, Sent = 995, SentPerMin = 989,41, WordPerSec = 907,42
Update = 19802, Epoch = 28, LR = 0,000426, AvgCost = 205,3591, Sent = 1000, SentPerMin = 988,81, WordPerSec = 907,50
Update = 19900, Epoch = 29, LR = 0,000425, AvgCost = 214,5308, Sent = 280, SentPerMin = 958,07, WordPerSec = 898,99
Update = 20000, Epoch = 29, LR = 0,000424, AvgCost = 205,8453, Sent = 588, SentPerMin = 985,08, WordPerSec = 901,09
Update = 20100, Epoch = 29, LR = 0,000423, AvgCost = 206,5281, Sent = 877, SentPerMin = 705,57, WordPerSec = 650,71
Update = 20140, Epoch = 29, LR = 0,000423, AvgCost = 205,3184, Sent = 1000, SentPerMin = 735,16, WordPerSec = 674,71
Update = 20200, Epoch = 30, LR = 0,000422, AvgCost = 208,2094, Sent = 173, SentPerMin = 866,06, WordPerSec = 796,81
Update = 20300, Epoch = 30, LR = 0,000421, AvgCost = 209,6378, Sent = 469, SentPerMin = 745,88, WordPerSec = 688,02
Update = 20400, Epoch = 30, LR = 0,000420, AvgCost = 208,6078, Sent = 766, SentPerMin = 723,22, WordPerSec = 667,34
Update = 20478, Epoch = 30, LR = 0,000419, AvgCost = 206,3125, Sent = 1000, SentPerMin = 722,88, WordPerSec = 663,44
Update = 20500, Epoch = 31, LR = 0,000419, AvgCost = 197,4774, Sent = 71, SentPerMin = 1026,43, WordPerSec = 888,37
Update = 20600, Epoch = 31, LR = 0,000418, AvgCost = 208,6039, Sent = 358, SentPerMin = 971,47, WordPerSec = 892,37
Update = 20700, Epoch = 31, LR = 0,000417, AvgCost = 206,1607, Sent = 652, SentPerMin = 978,33, WordPerSec = 898,38
Update = 20800, Epoch = 31, LR = 0,000416, AvgCost = 204,9813, Sent = 955, SentPerMin = 986,47, WordPerSec = 905,02
Update = 20816, Epoch = 31, LR = 0,000416, AvgCost = 205,0031, Sent = 1000, SentPerMin = 986,90, WordPerSec = 905,74
Update = 20900, Epoch = 32, LR = 0,000415, AvgCost = 206,3306, Sent = 244, SentPerMin = 987,21, WordPerSec = 901,91
Update = 21000, Epoch = 32, LR = 0,000414, AvgCost = 205,7421, Sent = 542, SentPerMin = 981,12, WordPerSec = 901,11
Update = 21100, Epoch = 32, LR = 0,000413, AvgCost = 205,4065, Sent = 835, SentPerMin = 984,42, WordPerSec = 905,69
Update = 21154, Epoch = 32, LR = 0,000413, AvgCost = 204,2899, Sent = 1000, SentPerMin = 988,16, WordPerSec = 906,90
Update = 21200, Epoch = 33, LR = 0,000412, AvgCost = 201,6446, Sent = 136, SentPerMin = 991,60, WordPerSec = 882,36
Update = 21300, Epoch = 33, LR = 0,000411, AvgCost = 205,6809, Sent = 428, SentPerMin = 963,31, WordPerSec = 886,04
Update = 21400, Epoch = 33, LR = 0,000410, AvgCost = 203,7382, Sent = 725, SentPerMin = 971,35, WordPerSec = 890,41
Update = 21492, Epoch = 33, LR = 0,000409, AvgCost = 203,5028, Sent = 1000, SentPerMin = 956,08, WordPerSec = 877,46
Update = 21500, Epoch = 34, LR = 0,000409, AvgCost = 188,2185, Sent = 23, SentPerMin = 907,96, WordPerSec = 791,50
Update = 21600, Epoch = 34, LR = 0,000408, AvgCost = 209,2027, Sent = 311, SentPerMin = 930,39, WordPerSec = 865,02
Update = 21700, Epoch = 34, LR = 0,000407, AvgCost = 204,5225, Sent = 611, SentPerMin = 940,45, WordPerSec = 866,19
Update = 21800, Epoch = 34, LR = 0,000406, AvgCost = 204,0428, Sent = 906, SentPerMin = 948,39, WordPerSec = 874,89
Update = 21830, Epoch = 34, LR = 0,000406, AvgCost = 202,6293, Sent = 1000, SentPerMin = 956,33, WordPerSec = 877,69
Update = 21900, Epoch = 35, LR = 0,000405, AvgCost = 206,4004, Sent = 200, SentPerMin = 914,32, WordPerSec = 839,04
Update = 22000, Epoch = 35, LR = 0,000405, AvgCost = 203,0120, Sent = 501, SentPerMin = 918,34, WordPerSec = 841,32
Update = 22100, Epoch = 35, LR = 0,000404, AvgCost = 203,8077, Sent = 795, SentPerMin = 921,02, WordPerSec = 848,63
Update = 22168, Epoch = 35, LR = 0,000403, AvgCost = 201,8455, Sent = 1000, SentPerMin = 940,49, WordPerSec = 863,15
Update = 22200, Epoch = 36, LR = 0,000403, AvgCost = 196,8755, Sent = 99, SentPerMin = 1013,27, WordPerSec = 882,94
Update = 22300, Epoch = 36, LR = 0,000402, AvgCost = 203,3147, Sent = 388, SentPerMin = 973,10, WordPerSec = 891,88
Update = 22400, Epoch = 36, LR = 0,000401, AvgCost = 201,7103, Sent = 683, SentPerMin = 968,78, WordPerSec = 887,69
Update = 22500, Epoch = 36, LR = 0,000400, AvgCost = 201,2276, Sent = 984, SentPerMin = 939,56, WordPerSec = 862,23
Update = 22506, Epoch = 36, LR = 0,000400, AvgCost = 201,1211, Sent = 1000, SentPerMin = 938,92, WordPerSec = 861,71
Update = 22600, Epoch = 37, LR = 0,000399, AvgCost = 209,3018, Sent = 269, SentPerMin = 881,86, WordPerSec = 824,27
Update = 22700, Epoch = 37, LR = 0,000398, AvgCost = 202,1541, Sent = 573, SentPerMin = 884,92, WordPerSec = 812,72
Update = 22800, Epoch = 37, LR = 0,000397, AvgCost = 201,6366, Sent = 865, SentPerMin = 886,92, WordPerSec = 816,89
Update = 22844, Epoch = 37, LR = 0,000397, AvgCost = 200,6233, Sent = 1000, SentPerMin = 893,38, WordPerSec = 819,91
Update = 22900, Epoch = 38, LR = 0,000396, AvgCost = 204,8554, Sent = 161, SentPerMin = 872,68, WordPerSec = 800,50
Update = 23000, Epoch = 38, LR = 0,000396, AvgCost = 205,0114, Sent = 456, SentPerMin = 877,60, WordPerSec = 810,88
Update = 23100, Epoch = 38, LR = 0,000395, AvgCost = 201,7814, Sent = 757, SentPerMin = 887,62, WordPerSec = 816,45
Update = 23182, Epoch = 38, LR = 0,000394, AvgCost = 200,4588, Sent = 1000, SentPerMin = 896,21, WordPerSec = 822,51
Update = 23200, Epoch = 39, LR = 0,000394, AvgCost = 190,5176, Sent = 59, SentPerMin = 704,96, WordPerSec = 589,06
Update = 23300, Epoch = 39, LR = 0,000393, AvgCost = 205,8359, Sent = 343, SentPerMin = 745,30, WordPerSec = 689,10
Update = 23400, Epoch = 39, LR = 0,000392, AvgCost = 201,4384, Sent = 640, SentPerMin = 794,59, WordPerSec = 730,18
Update = 23500, Epoch = 39, LR = 0,000391, AvgCost = 200,1438, Sent = 943, SentPerMin = 817,51, WordPerSec = 750,38
Update = 23520, Epoch = 39, LR = 0,000391, AvgCost = 200,0447, Sent = 1000, SentPerMin = 820,73, WordPerSec = 753,24
Update = 23600, Epoch = 40, LR = 0,000391, AvgCost = 203,9256, Sent = 231, SentPerMin = 715,64, WordPerSec = 658,53
Update = 23700, Epoch = 40, LR = 0,000390, AvgCost = 203,1409, Sent = 529, SentPerMin = 652,65, WordPerSec = 600,69
Update = 23800, Epoch = 40, LR = 0,000389, AvgCost = 202,0306, Sent = 824, SentPerMin = 641,48, WordPerSec = 590,28
Update = 23858, Epoch = 40, LR = 0,000388, AvgCost = 200,9514, Sent = 1000, SentPerMin = 640,46, WordPerSec = 587,80
Update = 23900, Epoch = 41, LR = 0,000388, AvgCost = 198,1531, Sent = 126, SentPerMin = 881,99, WordPerSec = 778,62
Update = 24000, Epoch = 41, LR = 0,000387, AvgCost = 201,6853, Sent = 418, SentPerMin = 810,12, WordPerSec = 743,39
Update = 24100, Epoch = 41, LR = 0,000386, AvgCost = 199,3206, Sent = 715, SentPerMin = 818,15, WordPerSec = 748,20
Update = 24196, Epoch = 41, LR = 0,000386, AvgCost = 199,7475, Sent = 1000, SentPerMin = 813,07, WordPerSec = 746,21
Update = 24200, Epoch = 42, LR = 0,000386, AvgCost = 180,2738, Sent = 12, SentPerMin = 812,76, WordPerSec = 731,49
Update = 24300, Epoch = 42, LR = 0,000385, AvgCost = 206,8129, Sent = 299, SentPerMin = 773,35, WordPerSec = 721,36
Update = 24400, Epoch = 42, LR = 0,000384, AvgCost = 200,3696, Sent = 601, SentPerMin = 775,19, WordPerSec = 712,12
Update = 24500, Epoch = 42, LR = 0,000383, AvgCost = 199,7417, Sent = 897, SentPerMin = 777,90, WordPerSec = 715,53
Update = 24534, Epoch = 42, LR = 0,000383, AvgCost = 198,8881, Sent = 1000, SentPerMin = 780,77, WordPerSec = 716,56
Update = 24600, Epoch = 43, LR = 0,000383, AvgCost = 204,6237, Sent = 189, SentPerMin = 737,36, WordPerSec = 679,23
Update = 24700, Epoch = 43, LR = 0,000382, AvgCost = 199,8026, Sent = 490, SentPerMin = 750,20, WordPerSec = 685,82
Update = 24800, Epoch = 43, LR = 0,000381, AvgCost = 200,5970, Sent = 784, SentPerMin = 742,29, WordPerSec = 684,11
Update = 24872, Epoch = 43, LR = 0,000380, AvgCost = 198,4103, Sent = 1000, SentPerMin = 750,98, WordPerSec = 689,22
Update = 24900, Epoch = 44, LR = 0,000380, AvgCost = 192,6996, Sent = 87, SentPerMin = 769,52, WordPerSec = 676,20
Update = 25000, Epoch = 44, LR = 0,000379, AvgCost = 201,4754, Sent = 375, SentPerMin = 734,73, WordPerSec = 677,19
Update = 25100, Epoch = 44, LR = 0,000379, AvgCost = 199,5252, Sent = 668, SentPerMin = 738,61, WordPerSec = 680,61
Update = 25200, Epoch = 44, LR = 0,000378, AvgCost = 196,8848, Sent = 974, SentPerMin = 752,35, WordPerSec = 688,46
Update = 25210, Epoch = 44, LR = 0,000378, AvgCost = 197,9290, Sent = 1000, SentPerMin = 749,99, WordPerSec = 688,32
Update = 25300, Epoch = 45, LR = 0,000377, AvgCost = 205,1786, Sent = 258, SentPerMin = 745,61, WordPerSec = 694,98
Update = 25400, Epoch = 45, LR = 0,000376, AvgCost = 198,0722, Sent = 562, SentPerMin = 751,94, WordPerSec = 689,48
Update = 25500, Epoch = 45, LR = 0,000376, AvgCost = 198,5244, Sent = 854, SentPerMin = 751,92, WordPerSec = 692,43
Update = 25548, Epoch = 45, LR = 0,000375, AvgCost = 197,3636, Sent = 1000, SentPerMin = 755,92, WordPerSec = 693,75
Update = 25600, Epoch = 46, LR = 0,000375, AvgCost = 196,4658, Sent = 152, SentPerMin = 736,68, WordPerSec = 662,77
Update = 25700, Epoch = 46, LR = 0,000374, AvgCost = 201,2017, Sent = 443, SentPerMin = 735,06, WordPerSec = 679,61
Update = 25800, Epoch = 46, LR = 0,000374, AvgCost = 197,2745, Sent = 744, SentPerMin = 744,80, WordPerSec = 684,05
Update = 25886, Epoch = 46, LR = 0,000373, AvgCost = 196,8432, Sent = 1000, SentPerMin = 748,89, WordPerSec = 687,31
Update = 25900, Epoch = 47, LR = 0,000373, AvgCost = 187,4085, Sent = 45, SentPerMin = 579,74, WordPerSec = 474,74
Update = 26000, Epoch = 47, LR = 0,000372, AvgCost = 203,0333, Sent = 330, SentPerMin = 635,37, WordPerSec = 588,97
Update = 26100, Epoch = 47, LR = 0,000371, AvgCost = 198,0311, Sent = 630, SentPerMin = 679,71, WordPerSec = 623,33
Update = 26200, Epoch = 47, LR = 0,000371, AvgCost = 197,4116, Sent = 929, SentPerMin = 698,89, WordPerSec = 642,10
Update = 26224, Epoch = 47, LR = 0,000371, AvgCost = 196,9405, Sent = 1000, SentPerMin = 703,63, WordPerSec = 645,77
Update = 26300, Epoch = 48, LR = 0,000370, AvgCost = 201,6558, Sent = 219, SentPerMin = 621,36, WordPerSec = 572,56
Update = 26400, Epoch = 48, LR = 0,000369, AvgCost = 199,4959, Sent = 519, SentPerMin = 574,53, WordPerSec = 527,21
Update = 26500, Epoch = 48, LR = 0,000369, AvgCost = 198,9768, Sent = 814, SentPerMin = 566,46, WordPerSec = 521,03
Update = 26562, Epoch = 48, LR = 0,000368, AvgCost = 197,8686, Sent = 1000, SentPerMin = 570,23, WordPerSec = 523,34
Update = 26600, Epoch = 49, LR = 0,000368, AvgCost = 184,4024, Sent = 119, SentPerMin = 807,82, WordPerSec = 687,21
Update = 26700, Epoch = 49, LR = 0,000367, AvgCost = 199,2025, Sent = 406, SentPerMin = 755,47, WordPerSec = 694,35
Update = 26800, Epoch = 49, LR = 0,000367, AvgCost = 196,6774, Sent = 702, SentPerMin = 783,08, WordPerSec = 716,15
Update = 26900, Epoch = 49, LR = 0,000366, AvgCost = 196,8193, Sent = 1000, SentPerMin = 787,31, WordPerSec = 722,56
Update = 26900, Epoch = 49, LR = 0,000366, AvgCost = 196,8193, Sent = 1000, SentPerMin = 787,30, WordPerSec = 722,56
Update = 27000, Epoch = 50, LR = 0,000365, AvgCost = 205,6623, Sent = 285, SentPerMin = 740,48, WordPerSec = 695,31
Update = 27100, Epoch = 50, LR = 0,000364, AvgCost = 196,8959, Sent = 592, SentPerMin = 788,28, WordPerSec = 723,10
Update = 27200, Epoch = 50, LR = 0,000364, AvgCost = 197,7522, Sent = 882, SentPerMin = 793,16, WordPerSec = 732,74
Update = 27238, Epoch = 50, LR = 0,000364, AvgCost = 196,1204, Sent = 1000, SentPerMin = 802,26, WordPerSec = 736,29
Update = 27300, Epoch = 51, LR = 0,000363, AvgCost = 198,2511, Sent = 178, SentPerMin = 790,04, WordPerSec = 729,01
Update = 27400, Epoch = 51, LR = 0,000362, AvgCost = 198,6179, Sent = 475, SentPerMin = 801,12, WordPerSec = 737,93
Update = 27500, Epoch = 51, LR = 0,000362, AvgCost = 197,9257, Sent = 771, SentPerMin = 804,92, WordPerSec = 743,98
Update = 27576, Epoch = 51, LR = 0,000361, AvgCost = 195,4257, Sent = 1000, SentPerMin = 819,67, WordPerSec = 752,26
Update = 27600, Epoch = 52, LR = 0,000361, AvgCost = 190,0927, Sent = 76, SentPerMin = 843,71, WordPerSec = 744,17
Update = 27700, Epoch = 52, LR = 0,000361, AvgCost = 198,7296, Sent = 363, SentPerMin = 849,47, WordPerSec = 783,32
Update = 27800, Epoch = 52, LR = 0,000360, AvgCost = 196,3069, Sent = 656, SentPerMin = 858,46, WordPerSec = 790,41
Update = 27900, Epoch = 52, LR = 0,000359, AvgCost = 194,3731, Sent = 961, SentPerMin = 871,01, WordPerSec = 798,33
Update = 27914, Epoch = 52, LR = 0,000359, AvgCost = 194,7855, Sent = 1000, SentPerMin = 871,09, WordPerSec = 799,45
Update = 28000, Epoch = 53, LR = 0,000359, AvgCost = 198,7279, Sent = 248, SentPerMin = 857,45, WordPerSec = 792,10
Update = 28100, Epoch = 53, LR = 0,000358, AvgCost = 195,5638, Sent = 549, SentPerMin = 869,21, WordPerSec = 796,41
Update = 28200, Epoch = 53, LR = 0,000357, AvgCost = 195,7058, Sent = 841, SentPerMin = 866,20, WordPerSec = 797,64
Update = 28252, Epoch = 53, LR = 0,000357, AvgCost = 194,5203, Sent = 1000, SentPerMin = 870,94, WordPerSec = 799,32
Update = 28300, Epoch = 54, LR = 0,000357, AvgCost = 192,7056, Sent = 140, SentPerMin = 856,32, WordPerSec = 768,65
Update = 28400, Epoch = 54, LR = 0,000356, AvgCost = 199,4530, Sent = 431, SentPerMin = 841,72, WordPerSec = 780,24
Update = 28500, Epoch = 54, LR = 0,000355, AvgCost = 193,7203, Sent = 732, SentPerMin = 861,77, WordPerSec = 787,88
Update = 28590, Epoch = 54, LR = 0,000355, AvgCost = 193,9852, Sent = 1000, SentPerMin = 865,33, WordPerSec = 794,17
Update = 28600, Epoch = 55, LR = 0,000355, AvgCost = 162,4258, Sent = 33, SentPerMin = 693,37, WordPerSec = 535,79
Update = 28700, Epoch = 55, LR = 0,000354, AvgCost = 200,9447, Sent = 317, SentPerMin = 723,00, WordPerSec = 673,28
Update = 28800, Epoch = 55, LR = 0,000354, AvgCost = 195,4631, Sent = 618, SentPerMin = 782,19, WordPerSec = 717,87
Update = 28900, Epoch = 55, LR = 0,000353, AvgCost = 194,9150, Sent = 914, SentPerMin = 802,89, WordPerSec = 739,86
Update = 28928, Epoch = 55, LR = 0,000353, AvgCost = 193,8391, Sent = 1000, SentPerMin = 810,63, WordPerSec = 743,97
Update = 29000, Epoch = 56, LR = 0,000352, AvgCost = 201,9939, Sent = 205, SentPerMin = 712,53, WordPerSec = 665,26
Update = 29100, Epoch = 56, LR = 0,000352, AvgCost = 196,1661, Sent = 506, SentPerMin = 655,37, WordPerSec = 601,12
Update = 29200, Epoch = 56, LR = 0,000351, AvgCost = 196,1833, Sent = 802, SentPerMin = 639,94, WordPerSec = 589,16
Update = 29266, Epoch = 56, LR = 0,000351, AvgCost = 194,4510, Sent = 1000, SentPerMin = 662,68, WordPerSec = 608,18
Update = 29300, Epoch = 57, LR = 0,000351, AvgCost = 185,8378, Sent = 106, SentPerMin = 896,32, WordPerSec = 772,02
Update = 29400, Epoch = 57, LR = 0,000350, AvgCost = 196,8658, Sent = 393, SentPerMin = 845,64, WordPerSec = 779,36
Update = 29500, Epoch = 57, LR = 0,000349, AvgCost = 194,2415, Sent = 689, SentPerMin = 857,84, WordPerSec = 786,35
Update = 29600, Epoch = 57, LR = 0,000349, AvgCost = 193,9684, Sent = 989, SentPerMin = 825,62, WordPerSec = 757,93
Update = 29604, Epoch = 57, LR = 0,000349, AvgCost = 193,8221, Sent = 1000, SentPerMin = 822,72, WordPerSec = 755,07
Update = 29700, Epoch = 58, LR = 0,000348, AvgCost = 203,2740, Sent = 275, SentPerMin = 827,41, WordPerSec = 776,72
Update = 29800, Epoch = 58, LR = 0,000348, AvgCost = 194,5572, Sent = 581, SentPerMin = 848,87, WordPerSec = 778,71
Update = 29900, Epoch = 58, LR = 0,000347, AvgCost = 194,6374, Sent = 870, SentPerMin = 851,98, WordPerSec = 786,61
Update = 29942, Epoch = 58, LR = 0,000347, AvgCost = 193,3068, Sent = 1000, SentPerMin = 859,93, WordPerSec = 789,21
Update = 30000, Epoch = 59, LR = 0,000346, AvgCost = 196,3475, Sent = 166, SentPerMin = 784,76, WordPerSec = 723,15
Update = 30100, Epoch = 59, LR = 0,000346, AvgCost = 196,6929, Sent = 463, SentPerMin = 440,62, WordPerSec = 406,70
Update = 30200, Epoch = 59, LR = 0,000345, AvgCost = 194,6564, Sent = 761, SentPerMin = 531,61, WordPerSec = 489,53
Update = 30280, Epoch = 59, LR = 0,000345, AvgCost = 192,7243, Sent = 1000, SentPerMin = 579,31, WordPerSec = 531,68
Update = 30300, Epoch = 60, LR = 0,000345, AvgCost = 188,1885, Sent = 64, SentPerMin = 833,70, WordPerSec = 726,89
Update = 30400, Epoch = 60, LR = 0,000344, AvgCost = 195,9391, Sent = 352, SentPerMin = 768,46, WordPerSec = 706,68
Update = 30500, Epoch = 60, LR = 0,000344, AvgCost = 193,7475, Sent = 645, SentPerMin = 763,54, WordPerSec = 702,81
Update = 30600, Epoch = 60, LR = 0,000343, AvgCost = 192,6758, Sent = 948, SentPerMin = 767,28, WordPerSec = 705,36
Update = 30618, Epoch = 60, LR = 0,000343, AvgCost = 192,2274, Sent = 1000, SentPerMin = 769,68, WordPerSec = 706,39
Update = 30700, Epoch = 61, LR = 0,000342, AvgCost = 194,1053, Sent = 239, SentPerMin = 776,00, WordPerSec = 708,68
Update = 30800, Epoch = 61, LR = 0,000342, AvgCost = 194,1780, Sent = 536, SentPerMin = 766,42, WordPerSec = 704,48
Update = 30900, Epoch = 61, LR = 0,000341, AvgCost = 192,6964, Sent = 831, SentPerMin = 767,41, WordPerSec = 704,66
Update = 30956, Epoch = 61, LR = 0,000341, AvgCost = 191,9538, Sent = 1000, SentPerMin = 775,99, WordPerSec = 712,18
Update = 31000, Epoch = 62, LR = 0,000341, AvgCost = 189,8914, Sent = 131, SentPerMin = 811,33, WordPerSec = 722,04
Update = 31100, Epoch = 62, LR = 0,000340, AvgCost = 193,6756, Sent = 423, SentPerMin = 781,01, WordPerSec = 717,74
Update = 31200, Epoch = 62, LR = 0,000340, AvgCost = 191,4085, Sent = 720, SentPerMin = 780,66, WordPerSec = 714,43
Update = 31294, Epoch = 62, LR = 0,000339, AvgCost = 191,5035, Sent = 1000, SentPerMin = 776,80, WordPerSec = 712,92
Update = 31300, Epoch = 63, LR = 0,000339, AvgCost = 192,0966, Sent = 16, SentPerMin = 498,52, WordPerSec = 486,58
Update = 31400, Epoch = 63, LR = 0,000339, AvgCost = 199,8082, Sent = 304, SentPerMin = 594,79, WordPerSec = 555,53
Update = 31500, Epoch = 63, LR = 0,000338, AvgCost = 193,4244, Sent = 606, SentPerMin = 646,23, WordPerSec = 593,92
Update = 31600, Epoch = 63, LR = 0,000338, AvgCost = 192,7017, Sent = 902, SentPerMin = 668,83, WordPerSec = 615,99
Update = 31632, Epoch = 63, LR = 0,000337, AvgCost = 191,5008, Sent = 1000, SentPerMin = 676,49, WordPerSec = 620,86
Update = 31700, Epoch = 64, LR = 0,000337, AvgCost = 196,9971, Sent = 195, SentPerMin = 625,30, WordPerSec = 574,75
Update = 31800, Epoch = 64, LR = 0,000336, AvgCost = 193,6246, Sent = 495, SentPerMin = 565,00, WordPerSec = 516,35
Update = 31900, Epoch = 64, LR = 0,000336, AvgCost = 194,2823, Sent = 789, SentPerMin = 552,08, WordPerSec = 509,06
Update = 31970, Epoch = 64, LR = 0,000336, AvgCost = 192,2316, Sent = 1000, SentPerMin = 552,86, WordPerSec = 507,40
Update = 32000, Epoch = 65, LR = 0,000335, AvgCost = 189,6128, Sent = 92, SentPerMin = 734,52, WordPerSec = 651,89
Update = 32100, Epoch = 65, LR = 0,000335, AvgCost = 194,3180, Sent = 381, SentPerMin = 717,44, WordPerSec = 660,20
Update = 32200, Epoch = 65, LR = 0,000334, AvgCost = 192,2595, Sent = 676, SentPerMin = 720,24, WordPerSec = 661,23
Update = 32300, Epoch = 65, LR = 0,000334, AvgCost = 190,1674, Sent = 979, SentPerMin = 726,84, WordPerSec = 665,54
Update = 32308, Epoch = 65, LR = 0,000334, AvgCost = 191,1341, Sent = 1000, SentPerMin = 725,20, WordPerSec = 665,57
Update = 32400, Epoch = 66, LR = 0,000333, AvgCost = 197,3483, Sent = 265, SentPerMin = 714,33, WordPerSec = 662,93
Update = 32500, Epoch = 66, LR = 0,000333, AvgCost = 192,4786, Sent = 566, SentPerMin = 722,09, WordPerSec = 664,60
Update = 32600, Epoch = 66, LR = 0,000332, AvgCost = 191,7371, Sent = 859, SentPerMin = 724,32, WordPerSec = 667,46
Update = 32646, Epoch = 66, LR = 0,000332, AvgCost = 190,5688, Sent = 1000, SentPerMin = 731,79, WordPerSec = 671,62
Update = 32700, Epoch = 67, LR = 0,000332, AvgCost = 192,4794, Sent = 157, SentPerMin = 725,45, WordPerSec = 659,91
Update = 32800, Epoch = 67, LR = 0,000331, AvgCost = 194,2679, Sent = 450, SentPerMin = 716,51, WordPerSec = 660,71
Update = 32900, Epoch = 67, LR = 0,000331, AvgCost = 190,2727, Sent = 752, SentPerMin = 724,89, WordPerSec = 664,60
Update = 32984, Epoch = 67, LR = 0,000330, AvgCost = 190,1834, Sent = 1000, SentPerMin = 729,50, WordPerSec = 669,51
Update = 33000, Epoch = 68, LR = 0,000330, AvgCost = 183,4589, Sent = 51, SentPerMin = 752,29, WordPerSec = 648,54
Update = 33100, Epoch = 68, LR = 0,000330, AvgCost = 195,1468, Sent = 336, SentPerMin = 694,85, WordPerSec = 643,77
Update = 33200, Epoch = 68, LR = 0,000329, AvgCost = 190,3000, Sent = 636, SentPerMin = 698,33, WordPerSec = 639,83
Update = 33300, Epoch = 68, LR = 0,000329, AvgCost = 190,1551, Sent = 935, SentPerMin = 710,32, WordPerSec = 652,96
Update = 33322, Epoch = 68, LR = 0,000329, AvgCost = 189,6999, Sent = 1000, SentPerMin = 715,09, WordPerSec = 656,29
Update = 33400, Epoch = 69, LR = 0,000328, AvgCost = 192,1393, Sent = 225, SentPerMin = 730,78, WordPerSec = 671,78
Update = 33500, Epoch = 69, LR = 0,000328, AvgCost = 191,7545, Sent = 524, SentPerMin = 736,41, WordPerSec = 677,59
Update = 33600, Epoch = 69, LR = 0,000327, AvgCost = 190,2218, Sent = 819, SentPerMin = 737,14, WordPerSec = 678,02
Update = 33660, Epoch = 69, LR = 0,000327, AvgCost = 189,1501, Sent = 1000, SentPerMin = 741,58, WordPerSec = 680,60
Update = 33700, Epoch = 70, LR = 0,000327, AvgCost = 186,8435, Sent = 122, SentPerMin = 540,23, WordPerSec = 471,52
Update = 33800, Epoch = 70, LR = 0,000326, AvgCost = 192,2211, Sent = 411, SentPerMin = 624,17, WordPerSec = 574,86
Update = 33900, Epoch = 70, LR = 0,000326, AvgCost = 188,9116, Sent = 709, SentPerMin = 667,69, WordPerSec = 609,90
Update = 33998, Epoch = 70, LR = 0,000325, AvgCost = 189,1019, Sent = 1000, SentPerMin = 684,13, WordPerSec = 627,87
Update = 34000, Epoch = 71, LR = 0,000325, AvgCost = 167,0326, Sent = 6, SentPerMin = 746,38, WordPerSec = 677,96
Update = 34100, Epoch = 71, LR = 0,000325, AvgCost = 197,9264, Sent = 292, SentPerMin = 579,54, WordPerSec = 542,22
Update = 34200, Epoch = 71, LR = 0,000324, AvgCost = 192,7762, Sent = 595, SentPerMin = 548,61, WordPerSec = 504,93
Update = 34300, Epoch = 71, LR = 0,000324, AvgCost = 191,7710, Sent = 890, SentPerMin = 542,32, WordPerSec = 499,92
Update = 34336, Epoch = 71, LR = 0,000324, AvgCost = 190,4663, Sent = 1000, SentPerMin = 543,57, WordPerSec = 498,87
Update = 34400, Epoch = 72, LR = 0,000323, AvgCost = 194,4730, Sent = 184, SentPerMin = 738,87, WordPerSec = 678,84
Update = 34500, Epoch = 72, LR = 0,000323, AvgCost = 192,3359, Sent = 482, SentPerMin = 742,55, WordPerSec = 682,68
Update = 34600, Epoch = 72, LR = 0,000323, AvgCost = 192,4554, Sent = 775, SentPerMin = 738,62, WordPerSec = 684,34
Update = 34674, Epoch = 72, LR = 0,000322, AvgCost = 189,4998, Sent = 1000, SentPerMin = 749,41, WordPerSec = 687,79
Update = 34700, Epoch = 73, LR = 0,000322, AvgCost = 183,5704, Sent = 81, SentPerMin = 741,91, WordPerSec = 658,56
Update = 34800, Epoch = 73, LR = 0,000322, AvgCost = 193,4773, Sent = 368, SentPerMin = 713,84, WordPerSec = 660,63
Update = 34900, Epoch = 73, LR = 0,000321, AvgCost = 190,2829, Sent = 663, SentPerMin = 728,32, WordPerSec = 669,16
Update = 35000, Epoch = 73, LR = 0,000321, AvgCost = 187,9850, Sent = 969, SentPerMin = 744,67, WordPerSec = 680,95
Update = 35012, Epoch = 73, LR = 0,000321, AvgCost = 188,8147, Sent = 1000, SentPerMin = 743,01, WordPerSec = 681,91
Update = 35100, Epoch = 74, LR = 0,000320, AvgCost = 196,9380, Sent = 251, SentPerMin = 651,95, WordPerSec = 609,05
Update = 35200, Epoch = 74, LR = 0,000320, AvgCost = 189,3764, Sent = 556, SentPerMin = 629,84, WordPerSec = 576,51
Update = 35300, Epoch = 74, LR = 0,000319, AvgCost = 189,6616, Sent = 847, SentPerMin = 615,01, WordPerSec = 566,86
Update = 35350, Epoch = 74, LR = 0,000319, AvgCost = 188,5154, Sent = 1000, SentPerMin = 614,69, WordPerSec = 564,14
Update = 35400, Epoch = 75, LR = 0,000319, AvgCost = 190,0176, Sent = 145, SentPerMin = 573,44, WordPerSec = 521,56
Update = 35500, Epoch = 75, LR = 0,000318, AvgCost = 193,0600, Sent = 437, SentPerMin = 581,70, WordPerSec = 539,06
Update = 35600, Epoch = 75, LR = 0,000318, AvgCost = 188,0586, Sent = 739, SentPerMin = 595,09, WordPerSec = 545,51
Update = 35688, Epoch = 75, LR = 0,000318, AvgCost = 187,9833, Sent = 1000, SentPerMin = 598,83, WordPerSec = 549,58
Update = 35700, Epoch = 76, LR = 0,000318, AvgCost = 152,0576, Sent = 41, SentPerMin = 672,86, WordPerSec = 518,05
Update = 35800, Epoch = 76, LR = 0,000317, AvgCost = 193,4208, Sent = 324, SentPerMin = 591,07, WordPerSec = 549,02
Update = 35900, Epoch = 76, LR = 0,000317, AvgCost = 188,8472, Sent = 624, SentPerMin = 606,87, WordPerSec = 556,35
Update = 36000, Epoch = 76, LR = 0,000316, AvgCost = 188,2725, Sent = 921, SentPerMin = 603,87, WordPerSec = 555,59
Update = 36026, Epoch = 76, LR = 0,000316, AvgCost = 187,5734, Sent = 1000, SentPerMin = 605,95, WordPerSec = 556,12
Update = 36100, Epoch = 77, LR = 0,000316, AvgCost = 193,0825, Sent = 213, SentPerMin = 493,40, WordPerSec = 455,18
Update = 36200, Epoch = 77, LR = 0,000315, AvgCost = 190,9125, Sent = 511, SentPerMin = 549,36, WordPerSec = 506,57
Update = 36300, Epoch = 77, LR = 0,000315, AvgCost = 189,2644, Sent = 808, SentPerMin = 569,42, WordPerSec = 524,51
Update = 36364, Epoch = 77, LR = 0,000315, AvgCost = 187,6503, Sent = 1000, SentPerMin = 576,92, WordPerSec = 529,48
Update = 36400, Epoch = 78, LR = 0,000314, AvgCost = 177,9267, Sent = 113, SentPerMin = 623,63, WordPerSec = 534,23
Update = 36500, Epoch = 78, LR = 0,000314, AvgCost = 191,5453, Sent = 399, SentPerMin = 480,88, WordPerSec = 442,63
Update = 36600, Epoch = 78, LR = 0,000314, AvgCost = 189,7680, Sent = 696, SentPerMin = 479,67, WordPerSec = 439,48
Update = 36700, Epoch = 78, LR = 0,000313, AvgCost = 189,1079, Sent = 995, SentPerMin = 485,24, WordPerSec = 445,03
Update = 36702, Epoch = 78, LR = 0,000313, AvgCost = 189,0571, Sent = 1000, SentPerMin = 484,98, WordPerSec = 445,09
Update = 36800, Epoch = 79, LR = 0,000313, AvgCost = 196,5071, Sent = 280, SentPerMin = 652,29, WordPerSec = 612,07
Update = 36900, Epoch = 79, LR = 0,000312, AvgCost = 188,6843, Sent = 588, SentPerMin = 702,05, WordPerSec = 642,19
Update = 37000, Epoch = 79, LR = 0,000312, AvgCost = 189,1245, Sent = 877, SentPerMin = 727,60, WordPerSec = 671,03
Update = 37040, Epoch = 79, LR = 0,000312, AvgCost = 188,0405, Sent = 1000, SentPerMin = 740,96, WordPerSec = 680,03
Starting inference...
Inference results:
在在 了了  了 了 了 了 了。
在
在在在 。
在
在


             */

            Console.ReadLine();
        }

        private static void CopyTrainingFiles(string[] sourceFiles, string trainFolderPath, string validFolderPath)
        {
            foreach (var fileName in sourceFiles)
            {
                string trainDest = Path.Combine(trainFolderPath, fileName);
                string validDest = Path.Combine(validFolderPath, fileName);

                if (!File.Exists(trainDest))
                {
                    File.Copy(fileName, trainDest);
                }

                if (!File.Exists(validDest))
                {
                    File.Copy(fileName, validDest);
                }
            }
        }

        private static Seq2SeqOptions CreateOptions(string trainFolderPath, string validFolderPath)
        {
            return new Seq2SeqOptions
            {
                Task = ModeEnums.Train,
                TrainCorpusPath = trainFolderPath,
                ValidCorpusPaths = validFolderPath,
                SrcLang = "ENU",
                TgtLang = "CHS",
                EncoderLayerDepth = 4,
                DecoderLayerDepth = 4,
                SrcEmbeddingDim = 64,
                TgtEmbeddingDim = 64,
                HiddenSize = 64,
                MultiHeadNum = 8,
                StartLearningRate = 0.0006f,
                WarmUpSteps = 10000,
                WeightsUpdateCount = 10000,
                UpdateNumToStepDownLearningRate = 10000,
                MaxTokenSizePerBatch = 128,
                ValMaxTokenSizePerBatch = 128,
                MaxSrcSentLength = 110,
                MaxTgtSentLength = 110,
                MaxValidSrcSentLength = 110,
                MaxValidTgtSentLength = 110,
                PaddingType = PaddingEnums.AllowPadding,
                TooLongSequence = TooLongSequence.Truncation,
                ProcessorType = ProcessorTypeEnums.CPU,
                MaxEpochNum = 80, // 73, // 66, // 50, // 40, // 26, // 13, // 3, 
                SharedEmbeddings = false,
                ModelFilePath = "seq2seq_test80.model",
                EncoderType = EncoderTypeEnums.BiLSTM,
                DecoderType = DecoderTypeEnums.AttentionLSTM
            };
        }

        private static void ReportTrainingProgress(CostEventArg ep)
        {
            TimeSpan ts = DateTime.Now - ep.StartDateTime;
            double sentPerMin = ts.TotalMinutes > 0 ? ep.ProcessedSentencesInTotal / ts.TotalMinutes : 0;
            double wordPerSec = ts.TotalSeconds > 0 ? ep.ProcessedWordsInTotal / ts.TotalSeconds : 0;

            Console.WriteLine($"Update = {ep.Update}, Epoch = {ep.Epoch}, LR = {ep.LearningRate:F6}, AvgCost = {ep.AvgCostInTotal:F4}, Sent = {ep.ProcessedSentencesInTotal}, SentPerMin = {sentPerMin:F}, WordPerSec = {wordPerSec:F}");
        }

        public static void Seq2Seq_StatusUpdateWatcher(object? sender, EventArgs e)
        {
            if (e is not CostEventArg ep)
            {
                throw new ArgumentNullException(nameof(e), "The input event argument is not a CostEventArg.");
            }

            ReportTrainingProgress(ep);
        }

        public static void Seq2Seq_EpochEndWatcher(object? sender, EventArgs e)
        {
            if (e is not CostEventArg ep)
            {
                throw new ArgumentNullException(nameof(e), "The input event argument is not a CostEventArg.");
            }

            ReportTrainingProgress(ep);
        }
    }
}
