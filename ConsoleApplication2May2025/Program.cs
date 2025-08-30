using Seq2SeqSharp;
using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Enums;
using Seq2SeqSharp.LearningRate;
using Seq2SeqSharp.Metrics;
using Seq2SeqSharp.Optimizer;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;

namespace ConsoleApplication2May2025
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
            options.ModelFilePath = "seq2seq_test.model.test";

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
                batchSize: 32,
                decodingOptions: decodingOptions,
                srcSpmPath: srcSpmPath,
                tgtSpmPath: tgtSpmPath);

            Console.WriteLine("Inference results:");
            foreach (var line in File.ReadLines(testOutputPath))
            {
                Console.WriteLine(line);
            }


            /*
             Update = 10100, Epoch = 0, LR = 0,000597, AvgCost = 266,1632, Sent = 285, SentPerMin = 155,17, WordPerSec = 145,70
Update = 10200, Epoch = 0, LR = 0,000594, AvgCost = 247,0216, Sent = 592, SentPerMin = 157,53, WordPerSec = 144,50
Update = 10300, Epoch = 0, LR = 0,000591, AvgCost = 244,1707, Sent = 882, SentPerMin = 159,54, WordPerSec = 147,39
Update = 10338, Epoch = 0, LR = 0,000590, AvgCost = 241,0866, Sent = 1000, SentPerMin = 161,39, WordPerSec = 148,11
Update = 10400, Epoch = 1, LR = 0,000588, AvgCost = 232,8194, Sent = 178, SentPerMin = 183,17, WordPerSec = 169,02
Update = 10500, Epoch = 1, LR = 0,000586, AvgCost = 233,3584, Sent = 475, SentPerMin = 183,25, WordPerSec = 168,80
Update = 10600, Epoch = 1, LR = 0,000583, AvgCost = 231,7539, Sent = 771, SentPerMin = 183,39, WordPerSec = 169,51
Update = 10676, Epoch = 1, LR = 0,000581, AvgCost = 228,5044, Sent = 1000, SentPerMin = 186,32, WordPerSec = 171,00
Update = 10700, Epoch = 2, LR = 0,000580, AvgCost = 219,2678, Sent = 76, SentPerMin = 162,80, WordPerSec = 143,59
Update = 10800, Epoch = 2, LR = 0,000577, AvgCost = 229,5921, Sent = 363, SentPerMin = 161,66, WordPerSec = 149,07
Update = 10900, Epoch = 2, LR = 0,000575, AvgCost = 226,6660, Sent = 656, SentPerMin = 163,11, WordPerSec = 150,18
Update = 11000, Epoch = 2, LR = 0,000572, AvgCost = 224,2538, Sent = 961, SentPerMin = 164,37, WordPerSec = 150,65
Update = 11014, Epoch = 2, LR = 0,000572, AvgCost = 224,6560, Sent = 1000, SentPerMin = 164,67, WordPerSec = 151,13
Starting inference...
Inference results:
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,


             */

            /*
             Update = 10100, Epoch = 0, LR = 0,000597, AvgCost = 266,0382, Sent = 285, SentPerMin = 159,05, WordPerSec = 149,35
Update = 10200, Epoch = 0, LR = 0,000594, AvgCost = 247,0345, Sent = 592, SentPerMin = 161,64, WordPerSec = 148,27
Update = 10300, Epoch = 0, LR = 0,000591, AvgCost = 244,0553, Sent = 882, SentPerMin = 162,38, WordPerSec = 150,01
Update = 10338, Epoch = 0, LR = 0,000590, AvgCost = 240,9711, Sent = 1000, SentPerMin = 163,95, WordPerSec = 150,47
Update = 10400, Epoch = 1, LR = 0,000588, AvgCost = 234,6782, Sent = 178, SentPerMin = 183,44, WordPerSec = 169,27
Update = 10500, Epoch = 1, LR = 0,000586, AvgCost = 233,9963, Sent = 475, SentPerMin = 183,45, WordPerSec = 168,98
Update = 10600, Epoch = 1, LR = 0,000583, AvgCost = 232,1540, Sent = 771, SentPerMin = 183,53, WordPerSec = 169,64
Update = 10676, Epoch = 1, LR = 0,000581, AvgCost = 228,8643, Sent = 1000, SentPerMin = 186,40, WordPerSec = 171,07
Update = 10700, Epoch = 2, LR = 0,000580, AvgCost = 219,5882, Sent = 76, SentPerMin = 162,08, WordPerSec = 142,95
Update = 10800, Epoch = 2, LR = 0,000577, AvgCost = 229,9682, Sent = 363, SentPerMin = 161,60, WordPerSec = 149,02
Update = 10900, Epoch = 2, LR = 0,000575, AvgCost = 226,8112, Sent = 656, SentPerMin = 162,90, WordPerSec = 149,99
Update = 11000, Epoch = 2, LR = 0,000572, AvgCost = 224,4085, Sent = 961, SentPerMin = 164,13, WordPerSec = 150,44
Update = 11014, Epoch = 2, LR = 0,000572, AvgCost = 224,8223, Sent = 1000, SentPerMin = 164,40, WordPerSec = 150,88
Update = 11100, Epoch = 3, LR = 0,000569, AvgCost = 228,1953, Sent = 248, SentPerMin = 186,22, WordPerSec = 172,03
Update = 11200, Epoch = 3, LR = 0,000567, AvgCost = 224,0252, Sent = 549, SentPerMin = 185,25, WordPerSec = 169,73
Update = 11300, Epoch = 3, LR = 0,000564, AvgCost = 224,0419, Sent = 841, SentPerMin = 185,65, WordPerSec = 170,95
Update = 11352, Epoch = 3, LR = 0,000563, AvgCost = 222,6659, Sent = 1000, SentPerMin = 186,83, WordPerSec = 171,46
Update = 11400, Epoch = 4, LR = 0,000562, AvgCost = 219,8333, Sent = 140, SentPerMin = 166,33, WordPerSec = 149,30
Update = 11500, Epoch = 4, LR = 0,000560, AvgCost = 226,9789, Sent = 431, SentPerMin = 161,49, WordPerSec = 149,69
Update = 11600, Epoch = 4, LR = 0,000557, AvgCost = 220,2597, Sent = 732, SentPerMin = 163,54, WordPerSec = 149,51
Update = 11690, Epoch = 4, LR = 0,000555, AvgCost = 220,5441, Sent = 1000, SentPerMin = 163,29, WordPerSec = 149,86
Update = 11700, Epoch = 5, LR = 0,000555, AvgCost = 183,0796, Sent = 33, SentPerMin = 203,55, WordPerSec = 157,29
Update = 11800, Epoch = 5, LR = 0,000552, AvgCost = 226,5101, Sent = 317, SentPerMin = 181,03, WordPerSec = 168,59
Update = 11900, Epoch = 5, LR = 0,000550, AvgCost = 220,2983, Sent = 618, SentPerMin = 182,19, WordPerSec = 167,21
Update = 12000, Epoch = 5, LR = 0,000548, AvgCost = 219,9578, Sent = 914, SentPerMin = 182,74, WordPerSec = 168,39
Update = 12028, Epoch = 5, LR = 0,000547, AvgCost = 218,8135, Sent = 1000, SentPerMin = 183,89, WordPerSec = 168,77
Update = 12100, Epoch = 6, LR = 0,000545, AvgCost = 226,5412, Sent = 205, SentPerMin = 159,34, WordPerSec = 148,77
Update = 12200, Epoch = 6, LR = 0,000543, AvgCost = 219,3759, Sent = 506, SentPerMin = 159,72, WordPerSec = 146,50
Update = 12300, Epoch = 6, LR = 0,000541, AvgCost = 219,4588, Sent = 802, SentPerMin = 160,55, WordPerSec = 147,81
Update = 12366, Epoch = 6, LR = 0,000540, AvgCost = 217,5631, Sent = 1000, SentPerMin = 162,09, WordPerSec = 148,76
Update = 12400, Epoch = 7, LR = 0,000539, AvgCost = 207,5125, Sent = 106, SentPerMin = 187,05, WordPerSec = 161,11
Update = 12500, Epoch = 7, LR = 0,000537, AvgCost = 220,3413, Sent = 393, SentPerMin = 180,45, WordPerSec = 166,31
Update = 12600, Epoch = 7, LR = 0,000535, AvgCost = 216,7979, Sent = 689, SentPerMin = 184,63, WordPerSec = 169,24
Update = 12700, Epoch = 7, LR = 0,000532, AvgCost = 216,4892, Sent = 989, SentPerMin = 184,49, WordPerSec = 169,37
Update = 12704, Epoch = 7, LR = 0,000532, AvgCost = 216,3353, Sent = 1000, SentPerMin = 184,64, WordPerSec = 169,46
Update = 12800, Epoch = 8, LR = 0,000530, AvgCost = 227,2276, Sent = 275, SentPerMin = 159,23, WordPerSec = 149,47
Update = 12900, Epoch = 8, LR = 0,000528, AvgCost = 216,7848, Sent = 581, SentPerMin = 160,07, WordPerSec = 146,85
Update = 13000, Epoch = 8, LR = 0,000526, AvgCost = 216,9502, Sent = 870, SentPerMin = 161,08, WordPerSec = 148,72
Update = 13042, Epoch = 8, LR = 0,000525, AvgCost = 215,4301, Sent = 1000, SentPerMin = 162,27, WordPerSec = 148,93
Update = 13100, Epoch = 9, LR = 0,000524, AvgCost = 219,5331, Sent = 166, SentPerMin = 181,37, WordPerSec = 167,13
Update = 13200, Epoch = 9, LR = 0,000522, AvgCost = 218,7632, Sent = 463, SentPerMin = 180,72, WordPerSec = 166,80
Update = 13300, Epoch = 9, LR = 0,000520, AvgCost = 216,2975, Sent = 761, SentPerMin = 181,66, WordPerSec = 167,28
Update = 13380, Epoch = 9, LR = 0,000519, AvgCost = 214,2300, Sent = 1000, SentPerMin = 184,27, WordPerSec = 169,11
Update = 13400, Epoch = 10, LR = 0,000518, AvgCost = 209,5824, Sent = 64, SentPerMin = 165,34, WordPerSec = 144,16
Update = 13500, Epoch = 10, LR = 0,000516, AvgCost = 218,1892, Sent = 352, SentPerMin = 159,84, WordPerSec = 146,99
Update = 13600, Epoch = 10, LR = 0,000514, AvgCost = 215,0250, Sent = 645, SentPerMin = 160,79, WordPerSec = 148,00
Update = 13700, Epoch = 10, LR = 0,000513, AvgCost = 214,1198, Sent = 948, SentPerMin = 161,56, WordPerSec = 148,52
Update = 13718, Epoch = 10, LR = 0,000512, AvgCost = 213,6586, Sent = 1000, SentPerMin = 162,26, WordPerSec = 148,92
Update = 13800, Epoch = 11, LR = 0,000511, AvgCost = 215,4475, Sent = 239, SentPerMin = 185,76, WordPerSec = 169,64
Update = 13900, Epoch = 11, LR = 0,000509, AvgCost = 215,0262, Sent = 536, SentPerMin = 181,76, WordPerSec = 167,07
Update = 14000, Epoch = 11, LR = 0,000507, AvgCost = 213,4743, Sent = 831, SentPerMin = 182,95, WordPerSec = 167,99
Update = 14056, Epoch = 11, LR = 0,000506, AvgCost = 212,7078, Sent = 1000, SentPerMin = 183,88, WordPerSec = 168,76
Update = 14100, Epoch = 12, LR = 0,000505, AvgCost = 210,8969, Sent = 131, SentPerMin = 164,19, WordPerSec = 146,12
Update = 14200, Epoch = 12, LR = 0,000504, AvgCost = 214,6242, Sent = 423, SentPerMin = 159,48, WordPerSec = 146,56
Update = 14300, Epoch = 12, LR = 0,000502, AvgCost = 211,8500, Sent = 720, SentPerMin = 161,82, WordPerSec = 148,10
Update = 14394, Epoch = 12, LR = 0,000500, AvgCost = 212,2199, Sent = 1000, SentPerMin = 161,77, WordPerSec = 148,47
Starting inference...
Inference results:
和在 和 和
和在 的 的 的 的 的 的 的 的 的 的 的的。
和在 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的的。
和在 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的
和在 的

             */

            /*
             Update = 10100, Epoch = 0, LR = 0,000597, AvgCost = 266,8568, Sent = 285, SentPerMin = 1012,92, WordPerSec = 951,13
Update = 10200, Epoch = 0, LR = 0,000594, AvgCost = 247,4984, Sent = 592, SentPerMin = 1080,84, WordPerSec = 991,47
Update = 10300, Epoch = 0, LR = 0,000591, AvgCost = 244,5287, Sent = 882, SentPerMin = 1095,18, WordPerSec = 1011,76
Update = 10338, Epoch = 0, LR = 0,000590, AvgCost = 241,3994, Sent = 1000, SentPerMin = 1107,92, WordPerSec = 1016,81
Update = 10400, Epoch = 1, LR = 0,000588, AvgCost = 234,0096, Sent = 178, SentPerMin = 1244,66, WordPerSec = 1148,51
Update = 10500, Epoch = 1, LR = 0,000586, AvgCost = 233,8527, Sent = 475, SentPerMin = 1261,49, WordPerSec = 1161,98
Update = 10600, Epoch = 1, LR = 0,000583, AvgCost = 232,1695, Sent = 771, SentPerMin = 1265,17, WordPerSec = 1169,39
Update = 10676, Epoch = 1, LR = 0,000581, AvgCost = 228,8762, Sent = 1000, SentPerMin = 1279,31, WordPerSec = 1174,11
Update = 10700, Epoch = 2, LR = 0,000580, AvgCost = 220,3182, Sent = 76, SentPerMin = 1299,86, WordPerSec = 1146,50
Update = 10800, Epoch = 2, LR = 0,000577, AvgCost = 230,3241, Sent = 363, SentPerMin = 1257,45, WordPerSec = 1159,53
Update = 10900, Epoch = 2, LR = 0,000575, AvgCost = 227,2425, Sent = 656, SentPerMin = 1268,47, WordPerSec = 1167,92
Update = 11000, Epoch = 2, LR = 0,000572, AvgCost = 224,7079, Sent = 961, SentPerMin = 1279,35, WordPerSec = 1172,60
Update = 11014, Epoch = 2, LR = 0,000572, AvgCost = 225,0954, Sent = 1000, SentPerMin = 1278,27, WordPerSec = 1173,15
Update = 11100, Epoch = 3, LR = 0,000569, AvgCost = 227,9627, Sent = 248, SentPerMin = 1257,83, WordPerSec = 1161,97
Update = 11200, Epoch = 3, LR = 0,000567, AvgCost = 224,1441, Sent = 549, SentPerMin = 1272,66, WordPerSec = 1166,07
Update = 11300, Epoch = 3, LR = 0,000564, AvgCost = 224,1803, Sent = 841, SentPerMin = 1273,11, WordPerSec = 1172,34
Update = 11352, Epoch = 3, LR = 0,000563, AvgCost = 222,7298, Sent = 1000, SentPerMin = 1281,46, WordPerSec = 1176,08
Update = 11400, Epoch = 4, LR = 0,000562, AvgCost = 220,3875, Sent = 140, SentPerMin = 1270,34, WordPerSec = 1140,28
Update = 11500, Epoch = 4, LR = 0,000560, AvgCost = 227,7834, Sent = 431, SentPerMin = 1246,99, WordPerSec = 1155,90
Update = 11600, Epoch = 4, LR = 0,000557, AvgCost = 220,8365, Sent = 732, SentPerMin = 1274,83, WordPerSec = 1165,52
Update = 11690, Epoch = 4, LR = 0,000555, AvgCost = 220,9780, Sent = 1000, SentPerMin = 1279,12, WordPerSec = 1173,94
Update = 11700, Epoch = 5, LR = 0,000555, AvgCost = 179,9121, Sent = 33, SentPerMin = 1400,28, WordPerSec = 1082,03
Update = 11800, Epoch = 5, LR = 0,000552, AvgCost = 226,6934, Sent = 317, SentPerMin = 1256,44, WordPerSec = 1170,04
Update = 11900, Epoch = 5, LR = 0,000550, AvgCost = 220,9366, Sent = 618, SentPerMin = 1259,38, WordPerSec = 1155,82
Update = 12000, Epoch = 5, LR = 0,000548, AvgCost = 220,3578, Sent = 914, SentPerMin = 1234,95, WordPerSec = 1138,00
Update = 12028, Epoch = 5, LR = 0,000547, AvgCost = 219,2399, Sent = 1000, SentPerMin = 1242,91, WordPerSec = 1140,70
Update = 12100, Epoch = 6, LR = 0,000545, AvgCost = 226,0143, Sent = 205, SentPerMin = 1200,93, WordPerSec = 1121,26
Update = 12200, Epoch = 6, LR = 0,000543, AvgCost = 219,6362, Sent = 506, SentPerMin = 1204,78, WordPerSec = 1105,06
Update = 12300, Epoch = 6, LR = 0,000541, AvgCost = 219,7246, Sent = 802, SentPerMin = 1197,06, WordPerSec = 1102,06
Update = 12366, Epoch = 6, LR = 0,000540, AvgCost = 217,9152, Sent = 1000, SentPerMin = 1205,28, WordPerSec = 1106,17
Update = 12400, Epoch = 7, LR = 0,000539, AvgCost = 205,8944, Sent = 106, SentPerMin = 1240,76, WordPerSec = 1068,69
Update = 12500, Epoch = 7, LR = 0,000537, AvgCost = 220,0341, Sent = 393, SentPerMin = 1194,68, WordPerSec = 1101,06
Update = 12600, Epoch = 7, LR = 0,000535, AvgCost = 217,0273, Sent = 689, SentPerMin = 1198,38, WordPerSec = 1098,52
Update = 12700, Epoch = 7, LR = 0,000532, AvgCost = 216,8474, Sent = 989, SentPerMin = 1200,97, WordPerSec = 1102,51
Update = 12704, Epoch = 7, LR = 0,000532, AvgCost = 216,6962, Sent = 1000, SentPerMin = 1201,59, WordPerSec = 1102,78
Update = 12800, Epoch = 8, LR = 0,000530, AvgCost = 225,8354, Sent = 275, SentPerMin = 1186,79, WordPerSec = 1114,07
Update = 12900, Epoch = 8, LR = 0,000528, AvgCost = 216,6016, Sent = 581, SentPerMin = 1217,14, WordPerSec = 1116,55
Update = 13000, Epoch = 8, LR = 0,000526, AvgCost = 217,0969, Sent = 870, SentPerMin = 1219,26, WordPerSec = 1125,72
Update = 13042, Epoch = 8, LR = 0,000525, AvgCost = 215,6601, Sent = 1000, SentPerMin = 1230,20, WordPerSec = 1129,04
Update = 13100, Epoch = 9, LR = 0,000524, AvgCost = 217,9625, Sent = 166, SentPerMin = 1207,57, WordPerSec = 1112,76
Update = 13200, Epoch = 9, LR = 0,000522, AvgCost = 218,7953, Sent = 463, SentPerMin = 1211,74, WordPerSec = 1118,44
Update = 13300, Epoch = 9, LR = 0,000520, AvgCost = 216,7108, Sent = 761, SentPerMin = 1221,71, WordPerSec = 1125,01
Update = 13380, Epoch = 9, LR = 0,000519, AvgCost = 214,7813, Sent = 1000, SentPerMin = 1236,02, WordPerSec = 1134,38
Update = 13400, Epoch = 10, LR = 0,000518, AvgCost = 206,3068, Sent = 64, SentPerMin = 1283,28, WordPerSec = 1118,86
Update = 13500, Epoch = 10, LR = 0,000516, AvgCost = 218,2815, Sent = 352, SentPerMin = 1213,46, WordPerSec = 1115,90
Update = 13600, Epoch = 10, LR = 0,000514, AvgCost = 215,6857, Sent = 645, SentPerMin = 1220,95, WordPerSec = 1123,84
Update = 13700, Epoch = 10, LR = 0,000513, AvgCost = 214,6245, Sent = 948, SentPerMin = 1241,74, WordPerSec = 1141,53
Update = 13718, Epoch = 10, LR = 0,000512, AvgCost = 214,1599, Sent = 1000, SentPerMin = 1245,76, WordPerSec = 1143,32
Update = 13800, Epoch = 11, LR = 0,000511, AvgCost = 215,3049, Sent = 239, SentPerMin = 1267,90, WordPerSec = 1157,91
Update = 13900, Epoch = 11, LR = 0,000509, AvgCost = 215,6272, Sent = 536, SentPerMin = 1265,05, WordPerSec = 1162,82
Update = 14000, Epoch = 11, LR = 0,000507, AvgCost = 214,1379, Sent = 831, SentPerMin = 1265,54, WordPerSec = 1162,06
Update = 14056, Epoch = 11, LR = 0,000506, AvgCost = 213,4358, Sent = 1000, SentPerMin = 1267,73, WordPerSec = 1163,48
Update = 14100, Epoch = 12, LR = 0,000505, AvgCost = 210,4420, Sent = 131, SentPerMin = 1241,81, WordPerSec = 1105,15
Update = 14200, Epoch = 12, LR = 0,000504, AvgCost = 215,1090, Sent = 423, SentPerMin = 1232,99, WordPerSec = 1133,11
Update = 14300, Epoch = 12, LR = 0,000502, AvgCost = 212,6243, Sent = 720, SentPerMin = 1245,88, WordPerSec = 1140,18
Update = 14394, Epoch = 12, LR = 0,000500, AvgCost = 212,8426, Sent = 1000, SentPerMin = 1225,20, WordPerSec = 1124,44
Update = 14400, Epoch = 13, LR = 0,000500, AvgCost = 215,7039, Sent = 16, SentPerMin = 880,96, WordPerSec = 859,86
Update = 14500, Epoch = 13, LR = 0,000498, AvgCost = 221,3402, Sent = 304, SentPerMin = 1019,98, WordPerSec = 952,65
Update = 14600, Epoch = 13, LR = 0,000497, AvgCost = 214,3166, Sent = 606, SentPerMin = 1042,05, WordPerSec = 957,70
Update = 14700, Epoch = 13, LR = 0,000495, AvgCost = 213,7820, Sent = 902, SentPerMin = 1034,60, WordPerSec = 952,85
Update = 14732, Epoch = 13, LR = 0,000494, AvgCost = 212,5649, Sent = 1000, SentPerMin = 1038,13, WordPerSec = 952,76
Update = 14800, Epoch = 14, LR = 0,000493, AvgCost = 218,3478, Sent = 195, SentPerMin = 1193,88, WordPerSec = 1097,35
Update = 14900, Epoch = 14, LR = 0,000492, AvgCost = 213,4871, Sent = 495, SentPerMin = 1208,03, WordPerSec = 1104,03
Update = 15000, Epoch = 14, LR = 0,000490, AvgCost = 214,2377, Sent = 789, SentPerMin = 1183,06, WordPerSec = 1090,87
Update = 15070, Epoch = 14, LR = 0,000489, AvgCost = 211,8460, Sent = 1000, SentPerMin = 1187,33, WordPerSec = 1089,70
Update = 15100, Epoch = 15, LR = 0,000488, AvgCost = 209,8829, Sent = 92, SentPerMin = 1164,67, WordPerSec = 1033,64
Update = 15200, Epoch = 15, LR = 0,000487, AvgCost = 214,9381, Sent = 381, SentPerMin = 1149,06, WordPerSec = 1057,37
Update = 15300, Epoch = 15, LR = 0,000485, AvgCost = 212,4354, Sent = 676, SentPerMin = 1162,29, WordPerSec = 1067,06
Update = 15400, Epoch = 15, LR = 0,000483, AvgCost = 210,1915, Sent = 979, SentPerMin = 1174,54, WordPerSec = 1075,49
Update = 15408, Epoch = 15, LR = 0,000483, AvgCost = 211,2510, Sent = 1000, SentPerMin = 1172,20, WordPerSec = 1075,81
Update = 15500, Epoch = 16, LR = 0,000482, AvgCost = 218,3550, Sent = 265, SentPerMin = 1151,13, WordPerSec = 1068,31
Update = 15600, Epoch = 16, LR = 0,000480, AvgCost = 212,7589, Sent = 566, SentPerMin = 1160,00, WordPerSec = 1067,64
Update = 15700, Epoch = 16, LR = 0,000479, AvgCost = 212,0808, Sent = 859, SentPerMin = 1161,85, WordPerSec = 1070,64
Update = 15746, Epoch = 16, LR = 0,000478, AvgCost = 210,7636, Sent = 1000, SentPerMin = 1172,68, WordPerSec = 1076,24
Update = 15800, Epoch = 17, LR = 0,000477, AvgCost = 212,9054, Sent = 157, SentPerMin = 1158,57, WordPerSec = 1053,90
Update = 15900, Epoch = 17, LR = 0,000476, AvgCost = 214,6321, Sent = 450, SentPerMin = 1156,10, WordPerSec = 1066,06
Update = 16000, Epoch = 17, LR = 0,000474, AvgCost = 210,3108, Sent = 752, SentPerMin = 1164,11, WordPerSec = 1067,28
Update = 16084, Epoch = 17, LR = 0,000473, AvgCost = 210,2669, Sent = 1000, SentPerMin = 1171,75, WordPerSec = 1075,39
Update = 16100, Epoch = 18, LR = 0,000473, AvgCost = 203,5542, Sent = 51, SentPerMin = 1236,04, WordPerSec = 1065,58
Update = 16200, Epoch = 18, LR = 0,000471, AvgCost = 215,7008, Sent = 336, SentPerMin = 1145,26, WordPerSec = 1061,07
Update = 16300, Epoch = 18, LR = 0,000470, AvgCost = 210,2687, Sent = 636, SentPerMin = 1165,49, WordPerSec = 1067,85
Update = 16400, Epoch = 18, LR = 0,000469, AvgCost = 210,1208, Sent = 935, SentPerMin = 1170,24, WordPerSec = 1075,74
Update = 16422, Epoch = 18, LR = 0,000468, AvgCost = 209,6565, Sent = 1000, SentPerMin = 1174,69, WordPerSec = 1078,09
Update = 16500, Epoch = 19, LR = 0,000467, AvgCost = 212,5988, Sent = 225, SentPerMin = 1158,75, WordPerSec = 1065,20
Update = 16600, Epoch = 19, LR = 0,000466, AvgCost = 211,7884, Sent = 524, SentPerMin = 1166,18, WordPerSec = 1073,04
Update = 16700, Epoch = 19, LR = 0,000464, AvgCost = 210,3265, Sent = 819, SentPerMin = 1171,59, WordPerSec = 1077,63
Update = 16760, Epoch = 19, LR = 0,000463, AvgCost = 209,0214, Sent = 1000, SentPerMin = 1178,56, WordPerSec = 1081,64
Update = 16800, Epoch = 20, LR = 0,000463, AvgCost = 204,5965, Sent = 122, SentPerMin = 1207,57, WordPerSec = 1053,99
Update = 16900, Epoch = 20, LR = 0,000462, AvgCost = 211,3351, Sent = 411, SentPerMin = 1159,00, WordPerSec = 1067,44
Update = 17000, Epoch = 20, LR = 0,000460, AvgCost = 207,9248, Sent = 709, SentPerMin = 1182,43, WordPerSec = 1080,09
Update = 17098, Epoch = 20, LR = 0,000459, AvgCost = 208,2451, Sent = 1000, SentPerMin = 1182,46, WordPerSec = 1085,22
Starting inference...
Inference results:
的, 的 的
的, 的 的 的 的 的。
的, 的 的
的, 的 的
的, 的 的
*/

            /*
             Update = 10100, Epoch = 0, LR = 0,000597, AvgCost = 267,6692, Sent = 285, SentPerMin = 152,08, WordPerSec = 142,80
Update = 10200, Epoch = 0, LR = 0,000594, AvgCost = 247,9845, Sent = 592, SentPerMin = 156,08, WordPerSec = 143,17
Update = 10300, Epoch = 0, LR = 0,000591, AvgCost = 244,8671, Sent = 882, SentPerMin = 157,82, WordPerSec = 145,80
Update = 10338, Epoch = 0, LR = 0,000590, AvgCost = 241,7178, Sent = 1000, SentPerMin = 159,60, WordPerSec = 146,48
Update = 10400, Epoch = 1, LR = 0,000588, AvgCost = 234,8189, Sent = 178, SentPerMin = 180,54, WordPerSec = 166,59
Update = 10500, Epoch = 1, LR = 0,000586, AvgCost = 234,0297, Sent = 475, SentPerMin = 179,07, WordPerSec = 164,94
Update = 10600, Epoch = 1, LR = 0,000583, AvgCost = 232,1719, Sent = 771, SentPerMin = 179,82, WordPerSec = 166,20
Update = 10676, Epoch = 1, LR = 0,000581, AvgCost = 228,8581, Sent = 1000, SentPerMin = 182,89, WordPerSec = 167,85
Update = 10700, Epoch = 2, LR = 0,000580, AvgCost = 219,6371, Sent = 76, SentPerMin = 161,12, WordPerSec = 142,11
Update = 10800, Epoch = 2, LR = 0,000577, AvgCost = 229,9832, Sent = 363, SentPerMin = 159,84, WordPerSec = 147,39
Update = 10900, Epoch = 2, LR = 0,000575, AvgCost = 226,9530, Sent = 656, SentPerMin = 161,28, WordPerSec = 148,50
Update = 11000, Epoch = 2, LR = 0,000572, AvgCost = 224,4136, Sent = 961, SentPerMin = 162,44, WordPerSec = 148,88
Update = 11014, Epoch = 2, LR = 0,000572, AvgCost = 224,8031, Sent = 1000, SentPerMin = 162,72, WordPerSec = 149,34
Update = 11100, Epoch = 3, LR = 0,000569, AvgCost = 227,7568, Sent = 248, SentPerMin = 184,33, WordPerSec = 170,28
Update = 11200, Epoch = 3, LR = 0,000567, AvgCost = 223,6869, Sent = 549, SentPerMin = 183,29, WordPerSec = 167,94
Update = 11300, Epoch = 3, LR = 0,000564, AvgCost = 223,6621, Sent = 841, SentPerMin = 183,70, WordPerSec = 169,16
Update = 11352, Epoch = 3, LR = 0,000563, AvgCost = 222,1763, Sent = 1000, SentPerMin = 184,96, WordPerSec = 169,75
Update = 11400, Epoch = 4, LR = 0,000562, AvgCost = 219,7552, Sent = 140, SentPerMin = 164,77, WordPerSec = 147,90
Update = 11500, Epoch = 4, LR = 0,000560, AvgCost = 227,0104, Sent = 431, SentPerMin = 159,95, WordPerSec = 148,26
Update = 11600, Epoch = 4, LR = 0,000557, AvgCost = 220,0415, Sent = 732, SentPerMin = 162,65, WordPerSec = 148,71
Update = 11690, Epoch = 4, LR = 0,000555, AvgCost = 220,1972, Sent = 1000, SentPerMin = 162,53, WordPerSec = 149,16
Update = 11700, Epoch = 5, LR = 0,000555, AvgCost = 181,7975, Sent = 33, SentPerMin = 203,24, WordPerSec = 157,05
Update = 11800, Epoch = 5, LR = 0,000552, AvgCost = 226,3486, Sent = 317, SentPerMin = 180,98, WordPerSec = 168,53
Update = 11900, Epoch = 5, LR = 0,000550, AvgCost = 220,0777, Sent = 618, SentPerMin = 182,48, WordPerSec = 167,48
Update = 12000, Epoch = 5, LR = 0,000548, AvgCost = 219,5078, Sent = 914, SentPerMin = 183,15, WordPerSec = 168,77
Update = 12028, Epoch = 5, LR = 0,000547, AvgCost = 218,3120, Sent = 1000, SentPerMin = 184,28, WordPerSec = 169,13
Update = 12100, Epoch = 6, LR = 0,000545, AvgCost = 225,8395, Sent = 205, SentPerMin = 159,80, WordPerSec = 149,20
Update = 12200, Epoch = 6, LR = 0,000543, AvgCost = 218,9219, Sent = 506, SentPerMin = 160,29, WordPerSec = 147,02
Update = 12300, Epoch = 6, LR = 0,000541, AvgCost = 218,8017, Sent = 802, SentPerMin = 161,05, WordPerSec = 148,27
Update = 12366, Epoch = 6, LR = 0,000540, AvgCost = 216,9009, Sent = 1000, SentPerMin = 162,50, WordPerSec = 149,14
Update = 12400, Epoch = 7, LR = 0,000539, AvgCost = 206,4413, Sent = 106, SentPerMin = 187,39, WordPerSec = 161,40
Update = 12500, Epoch = 7, LR = 0,000537, AvgCost = 219,2488, Sent = 393, SentPerMin = 180,57, WordPerSec = 166,42
Update = 12600, Epoch = 7, LR = 0,000535, AvgCost = 215,8949, Sent = 689, SentPerMin = 184,66, WordPerSec = 169,27
Update = 12700, Epoch = 7, LR = 0,000532, AvgCost = 215,5224, Sent = 989, SentPerMin = 184,58, WordPerSec = 169,45
Update = 12704, Epoch = 7, LR = 0,000532, AvgCost = 215,3689, Sent = 1000, SentPerMin = 184,73, WordPerSec = 169,54
Update = 12800, Epoch = 8, LR = 0,000530, AvgCost = 225,5095, Sent = 275, SentPerMin = 159,49, WordPerSec = 149,72
Update = 12900, Epoch = 8, LR = 0,000528, AvgCost = 215,4474, Sent = 581, SentPerMin = 160,56, WordPerSec = 147,29
Update = 13000, Epoch = 8, LR = 0,000526, AvgCost = 215,8426, Sent = 870, SentPerMin = 161,41, WordPerSec = 149,03
Update = 13042, Epoch = 8, LR = 0,000525, AvgCost = 214,4756, Sent = 1000, SentPerMin = 162,61, WordPerSec = 149,24
Update = 13100, Epoch = 9, LR = 0,000524, AvgCost = 218,1321, Sent = 166, SentPerMin = 182,23, WordPerSec = 167,92
Update = 13200, Epoch = 9, LR = 0,000522, AvgCost = 217,6798, Sent = 463, SentPerMin = 181,11, WordPerSec = 167,17
Update = 13300, Epoch = 9, LR = 0,000520, AvgCost = 215,2971, Sent = 761, SentPerMin = 182,02, WordPerSec = 167,62
Update = 13380, Epoch = 9, LR = 0,000519, AvgCost = 213,4229, Sent = 1000, SentPerMin = 184,62, WordPerSec = 169,44
Update = 13400, Epoch = 10, LR = 0,000518, AvgCost = 208,5312, Sent = 64, SentPerMin = 165,60, WordPerSec = 144,38
Update = 13500, Epoch = 10, LR = 0,000516, AvgCost = 217,2312, Sent = 352, SentPerMin = 160,09, WordPerSec = 147,21
Update = 13600, Epoch = 10, LR = 0,000514, AvgCost = 214,2450, Sent = 645, SentPerMin = 160,98, WordPerSec = 148,17
Update = 13700, Epoch = 10, LR = 0,000513, AvgCost = 213,3473, Sent = 948, SentPerMin = 161,81, WordPerSec = 148,75
Update = 13718, Epoch = 10, LR = 0,000512, AvgCost = 212,9377, Sent = 1000, SentPerMin = 162,39, WordPerSec = 149,04
Update = 13800, Epoch = 11, LR = 0,000511, AvgCost = 214,3751, Sent = 239, SentPerMin = 181,62, WordPerSec = 165,87
Update = 13900, Epoch = 11, LR = 0,000509, AvgCost = 214,2261, Sent = 536, SentPerMin = 178,54, WordPerSec = 164,11
Update = 14000, Epoch = 11, LR = 0,000507, AvgCost = 212,6877, Sent = 831, SentPerMin = 179,58, WordPerSec = 164,89
Update = 14056, Epoch = 11, LR = 0,000506, AvgCost = 212,0248, Sent = 1000, SentPerMin = 180,58, WordPerSec = 165,73
Update = 14100, Epoch = 12, LR = 0,000505, AvgCost = 209,9931, Sent = 131, SentPerMin = 161,87, WordPerSec = 144,05
Update = 14200, Epoch = 12, LR = 0,000504, AvgCost = 213,9956, Sent = 423, SentPerMin = 157,11, WordPerSec = 144,39
Update = 14300, Epoch = 12, LR = 0,000502, AvgCost = 211,2400, Sent = 720, SentPerMin = 160,19, WordPerSec = 146,60
Update = 14394, Epoch = 12, LR = 0,000500, AvgCost = 211,5446, Sent = 1000, SentPerMin = 160,36, WordPerSec = 147,18
Update = 14400, Epoch = 13, LR = 0,000500, AvgCost = 210,3222, Sent = 16, SentPerMin = 172,97, WordPerSec = 168,83
Update = 14500, Epoch = 13, LR = 0,000498, AvgCost = 218,8705, Sent = 304, SentPerMin = 175,84, WordPerSec = 164,23
Update = 14600, Epoch = 13, LR = 0,000497, AvgCost = 212,1382, Sent = 606, SentPerMin = 176,15, WordPerSec = 161,90
Update = 14700, Epoch = 13, LR = 0,000495, AvgCost = 211,8232, Sent = 902, SentPerMin = 176,72, WordPerSec = 162,76
Update = 14732, Epoch = 13, LR = 0,000494, AvgCost = 210,6618, Sent = 1000, SentPerMin = 177,09, WordPerSec = 162,53
Update = 14800, Epoch = 14, LR = 0,000493, AvgCost = 215,7751, Sent = 195, SentPerMin = 147,81, WordPerSec = 135,86
Update = 14900, Epoch = 14, LR = 0,000492, AvgCost = 211,6985, Sent = 495, SentPerMin = 151,24, WordPerSec = 138,22
Update = 15000, Epoch = 14, LR = 0,000490, AvgCost = 212,4555, Sent = 789, SentPerMin = 150,89, WordPerSec = 139,13
Update = 15070, Epoch = 14, LR = 0,000489, AvgCost = 210,1248, Sent = 1000, SentPerMin = 153,61, WordPerSec = 140,98
Update = 15100, Epoch = 15, LR = 0,000488, AvgCost = 206,3862, Sent = 92, SentPerMin = 181,86, WordPerSec = 161,40
Update = 15200, Epoch = 15, LR = 0,000487, AvgCost = 212,9918, Sent = 381, SentPerMin = 180,51, WordPerSec = 166,11
Update = 15300, Epoch = 15, LR = 0,000485, AvgCost = 210,3703, Sent = 676, SentPerMin = 182,80, WordPerSec = 167,82
Update = 15400, Epoch = 15, LR = 0,000483, AvgCost = 208,1677, Sent = 979, SentPerMin = 182,89, WordPerSec = 167,46
Update = 15408, Epoch = 15, LR = 0,000483, AvgCost = 209,1996, Sent = 1000, SentPerMin = 182,59, WordPerSec = 167,58
Update = 15500, Epoch = 16, LR = 0,000482, AvgCost = 216,4431, Sent = 265, SentPerMin = 158,96, WordPerSec = 147,52
Update = 15600, Epoch = 16, LR = 0,000480, AvgCost = 211,4097, Sent = 566, SentPerMin = 159,05, WordPerSec = 146,39
Update = 15700, Epoch = 16, LR = 0,000479, AvgCost = 210,5528, Sent = 859, SentPerMin = 159,95, WordPerSec = 147,39
Update = 15746, Epoch = 16, LR = 0,000478, AvgCost = 209,0706, Sent = 1000, SentPerMin = 161,42, WordPerSec = 148,15
Update = 15800, Epoch = 17, LR = 0,000477, AvgCost = 210,1934, Sent = 157, SentPerMin = 182,37, WordPerSec = 165,90
Update = 15900, Epoch = 17, LR = 0,000476, AvgCost = 213,1879, Sent = 450, SentPerMin = 180,83, WordPerSec = 166,74
Update = 16000, Epoch = 17, LR = 0,000474, AvgCost = 208,5165, Sent = 752, SentPerMin = 181,76, WordPerSec = 166,64
Update = 16084, Epoch = 17, LR = 0,000473, AvgCost = 208,3016, Sent = 1000, SentPerMin = 183,67, WordPerSec = 168,56
Update = 16100, Epoch = 18, LR = 0,000473, AvgCost = 200,5675, Sent = 51, SentPerMin = 169,08, WordPerSec = 145,76
Update = 16200, Epoch = 18, LR = 0,000471, AvgCost = 214,3377, Sent = 336, SentPerMin = 159,96, WordPerSec = 148,20
Update = 16300, Epoch = 18, LR = 0,000470, AvgCost = 208,7622, Sent = 636, SentPerMin = 160,70, WordPerSec = 147,23
Update = 16400, Epoch = 18, LR = 0,000469, AvgCost = 208,4375, Sent = 935, SentPerMin = 161,08, WordPerSec = 148,07
Update = 16422, Epoch = 18, LR = 0,000468, AvgCost = 207,9570, Sent = 1000, SentPerMin = 162,05, WordPerSec = 148,73
Update = 16500, Epoch = 19, LR = 0,000467, AvgCost = 210,5695, Sent = 225, SentPerMin = 184,51, WordPerSec = 169,61
Update = 16600, Epoch = 19, LR = 0,000466, AvgCost = 210,0529, Sent = 524, SentPerMin = 180,90, WordPerSec = 166,45
Update = 16700, Epoch = 19, LR = 0,000464, AvgCost = 208,3112, Sent = 819, SentPerMin = 182,18, WordPerSec = 167,57
Update = 16760, Epoch = 19, LR = 0,000463, AvgCost = 207,1175, Sent = 1000, SentPerMin = 183,64, WordPerSec = 168,54
Update = 16800, Epoch = 20, LR = 0,000463, AvgCost = 202,7617, Sent = 122, SentPerMin = 165,54, WordPerSec = 144,48
Update = 16900, Epoch = 20, LR = 0,000462, AvgCost = 210,0893, Sent = 411, SentPerMin = 158,77, WordPerSec = 146,23
Update = 17000, Epoch = 20, LR = 0,000460, AvgCost = 206,2595, Sent = 709, SentPerMin = 162,13, WordPerSec = 148,10
Update = 17098, Epoch = 20, LR = 0,000459, AvgCost = 206,6893, Sent = 1000, SentPerMin = 161,73, WordPerSec = 148,43
Update = 17100, Epoch = 21, LR = 0,000459, AvgCost = 180,3571, Sent = 6, SentPerMin = 201,24, WordPerSec = 182,79
Update = 17200, Epoch = 21, LR = 0,000457, AvgCost = 214,8780, Sent = 292, SentPerMin = 180,64, WordPerSec = 169,01
Update = 17300, Epoch = 21, LR = 0,000456, AvgCost = 208,1181, Sent = 595, SentPerMin = 181,10, WordPerSec = 166,68
Update = 17400, Epoch = 21, LR = 0,000455, AvgCost = 207,2472, Sent = 890, SentPerMin = 182,52, WordPerSec = 168,25
Update = 17436, Epoch = 21, LR = 0,000454, AvgCost = 205,9089, Sent = 1000, SentPerMin = 183,88, WordPerSec = 168,76
Update = 17500, Epoch = 22, LR = 0,000454, AvgCost = 211,5848, Sent = 184, SentPerMin = 160,78, WordPerSec = 147,72
Update = 17600, Epoch = 22, LR = 0,000452, AvgCost = 208,3961, Sent = 482, SentPerMin = 159,36, WordPerSec = 146,51
Update = 17700, Epoch = 22, LR = 0,000451, AvgCost = 208,5995, Sent = 775, SentPerMin = 158,81, WordPerSec = 147,14
Update = 17774, Epoch = 22, LR = 0,000450, AvgCost = 205,6352, Sent = 1000, SentPerMin = 161,49, WordPerSec = 148,21
Update = 17800, Epoch = 23, LR = 0,000450, AvgCost = 199,9390, Sent = 81, SentPerMin = 181,13, WordPerSec = 160,78
Update = 17900, Epoch = 23, LR = 0,000448, AvgCost = 210,1095, Sent = 368, SentPerMin = 178,94, WordPerSec = 165,60
Update = 18000, Epoch = 23, LR = 0,000447, AvgCost = 206,0878, Sent = 663, SentPerMin = 182,25, WordPerSec = 167,45
Update = 18100, Epoch = 23, LR = 0,000446, AvgCost = 204,0283, Sent = 969, SentPerMin = 183,80, WordPerSec = 168,07
Update = 18112, Epoch = 23, LR = 0,000446, AvgCost = 204,8946, Sent = 1000, SentPerMin = 183,69, WordPerSec = 168,59
Update = 18200, Epoch = 24, LR = 0,000445, AvgCost = 214,1151, Sent = 251, SentPerMin = 160,27, WordPerSec = 149,73
Update = 18300, Epoch = 24, LR = 0,000444, AvgCost = 205,2213, Sent = 556, SentPerMin = 161,20, WordPerSec = 147,55
Update = 18400, Epoch = 24, LR = 0,000442, AvgCost = 205,8759, Sent = 847, SentPerMin = 160,79, WordPerSec = 148,20
Update = 18450, Epoch = 24, LR = 0,000442, AvgCost = 204,6220, Sent = 1000, SentPerMin = 161,93, WordPerSec = 148,62
Update = 18500, Epoch = 25, LR = 0,000441, AvgCost = 206,7077, Sent = 145, SentPerMin = 181,98, WordPerSec = 165,52
Update = 18600, Epoch = 25, LR = 0,000440, AvgCost = 209,5933, Sent = 437, SentPerMin = 180,62, WordPerSec = 167,38
Update = 18700, Epoch = 25, LR = 0,000439, AvgCost = 203,6315, Sent = 739, SentPerMin = 182,33, WordPerSec = 167,14
Update = 18788, Epoch = 25, LR = 0,000438, AvgCost = 203,8363, Sent = 1000, SentPerMin = 183,89, WordPerSec = 168,76
Starting inference...
Inference results:
和和•• (•• () 和和•• (•• () 和和•• (•• () 和和•• (•• () 和和•• (•• () 和和•• (•• () 和和
和, 和 和 和 和 和 和 和 和 和 和 和 和 和 和 和 和 和 和 和 和 和 和 和 和 和 和 和 和 和 和
和和•• (•• (•• () 和 和 和 和 和 和 和 和 和 和 和 和 和 和 和 和 和 和 和 和 和 和 和 和 和
和和• (•• (• () 和 和 和 和 和 和 和 和 和 和 和 和 和 和 和 和 和 和 和 和 和 和 和 和 和 和
和和•• (• () 的。

             */

            /*
             Update = 10100, Epoch = 0, LR = 0,000597, AvgCost = 266,5920, Sent = 285, SentPerMin = 152,01, WordPerSec = 142,74
Update = 10200, Epoch = 0, LR = 0,000594, AvgCost = 247,4447, Sent = 592, SentPerMin = 155,57, WordPerSec = 142,71
Update = 10300, Epoch = 0, LR = 0,000591, AvgCost = 244,4681, Sent = 882, SentPerMin = 157,16, WordPerSec = 145,19
Update = 10338, Epoch = 0, LR = 0,000590, AvgCost = 241,3655, Sent = 1000, SentPerMin = 159,03, WordPerSec = 145,95
Update = 10400, Epoch = 1, LR = 0,000588, AvgCost = 234,1522, Sent = 178, SentPerMin = 181,32, WordPerSec = 167,31
Update = 10500, Epoch = 1, LR = 0,000586, AvgCost = 233,9379, Sent = 475, SentPerMin = 181,04, WordPerSec = 166,76
Update = 10600, Epoch = 1, LR = 0,000583, AvgCost = 232,1435, Sent = 771, SentPerMin = 180,97, WordPerSec = 167,27
Update = 10676, Epoch = 1, LR = 0,000581, AvgCost = 228,8501, Sent = 1000, SentPerMin = 183,70, WordPerSec = 168,60
Update = 10700, Epoch = 2, LR = 0,000580, AvgCost = 219,5462, Sent = 76, SentPerMin = 160,64, WordPerSec = 141,69
Update = 10800, Epoch = 2, LR = 0,000577, AvgCost = 230,1787, Sent = 363, SentPerMin = 159,76, WordPerSec = 147,32
Update = 10900, Epoch = 2, LR = 0,000575, AvgCost = 226,9980, Sent = 656, SentPerMin = 161,08, WordPerSec = 148,32
Update = 11000, Epoch = 2, LR = 0,000572, AvgCost = 224,4109, Sent = 961, SentPerMin = 162,33, WordPerSec = 148,78
Update = 11014, Epoch = 2, LR = 0,000572, AvgCost = 224,8332, Sent = 1000, SentPerMin = 162,64, WordPerSec = 149,27
Update = 11100, Epoch = 3, LR = 0,000569, AvgCost = 227,5951, Sent = 248, SentPerMin = 183,23, WordPerSec = 169,26
Update = 11200, Epoch = 3, LR = 0,000567, AvgCost = 223,9728, Sent = 549, SentPerMin = 182,18, WordPerSec = 166,92
Update = 11300, Epoch = 3, LR = 0,000564, AvgCost = 223,9548, Sent = 841, SentPerMin = 182,77, WordPerSec = 168,30
Update = 11352, Epoch = 3, LR = 0,000563, AvgCost = 222,5699, Sent = 1000, SentPerMin = 184,09, WordPerSec = 168,95
Update = 11400, Epoch = 4, LR = 0,000562, AvgCost = 219,6303, Sent = 140, SentPerMin = 164,50, WordPerSec = 147,66
Update = 11500, Epoch = 4, LR = 0,000560, AvgCost = 227,3982, Sent = 431, SentPerMin = 159,53, WordPerSec = 147,87
Update = 11600, Epoch = 4, LR = 0,000557, AvgCost = 220,4874, Sent = 732, SentPerMin = 162,41, WordPerSec = 148,49
Update = 11690, Epoch = 4, LR = 0,000555, AvgCost = 220,7554, Sent = 1000, SentPerMin = 162,55, WordPerSec = 149,18
Update = 11700, Epoch = 5, LR = 0,000555, AvgCost = 183,1181, Sent = 33, SentPerMin = 201,72, WordPerSec = 155,88
Update = 11800, Epoch = 5, LR = 0,000552, AvgCost = 226,6829, Sent = 317, SentPerMin = 180,33, WordPerSec = 167,93
Update = 11900, Epoch = 5, LR = 0,000550, AvgCost = 220,3942, Sent = 618, SentPerMin = 178,57, WordPerSec = 163,88
Update = 12000, Epoch = 5, LR = 0,000548, AvgCost = 219,9914, Sent = 914, SentPerMin = 178,02, WordPerSec = 164,04
Update = 12028, Epoch = 5, LR = 0,000547, AvgCost = 218,8499, Sent = 1000, SentPerMin = 178,96, WordPerSec = 164,24
Update = 12100, Epoch = 6, LR = 0,000545, AvgCost = 226,3486, Sent = 205, SentPerMin = 153,39, WordPerSec = 143,22
Update = 12200, Epoch = 6, LR = 0,000543, AvgCost = 219,3270, Sent = 506, SentPerMin = 154,95, WordPerSec = 142,13
Update = 12300, Epoch = 6, LR = 0,000541, AvgCost = 219,0277, Sent = 802, SentPerMin = 157,70, WordPerSec = 145,19
Update = 12366, Epoch = 6, LR = 0,000540, AvgCost = 217,1705, Sent = 1000, SentPerMin = 159,80, WordPerSec = 146,66
Update = 12400, Epoch = 7, LR = 0,000539, AvgCost = 207,6455, Sent = 106, SentPerMin = 187,08, WordPerSec = 161,13
Update = 12500, Epoch = 7, LR = 0,000537, AvgCost = 219,9975, Sent = 393, SentPerMin = 180,22, WordPerSec = 166,10
Update = 12600, Epoch = 7, LR = 0,000535, AvgCost = 216,5033, Sent = 689, SentPerMin = 184,54, WordPerSec = 169,16
Update = 12700, Epoch = 7, LR = 0,000532, AvgCost = 216,1110, Sent = 989, SentPerMin = 184,24, WordPerSec = 169,14
Update = 12704, Epoch = 7, LR = 0,000532, AvgCost = 215,9665, Sent = 1000, SentPerMin = 184,38, WordPerSec = 169,22
Update = 12800, Epoch = 8, LR = 0,000530, AvgCost = 226,2401, Sent = 275, SentPerMin = 159,51, WordPerSec = 149,74
Update = 12900, Epoch = 8, LR = 0,000528, AvgCost = 216,2488, Sent = 581, SentPerMin = 160,51, WordPerSec = 147,25
Update = 13000, Epoch = 8, LR = 0,000526, AvgCost = 216,4856, Sent = 870, SentPerMin = 161,55, WordPerSec = 149,15
Update = 13042, Epoch = 8, LR = 0,000525, AvgCost = 215,1182, Sent = 1000, SentPerMin = 162,75, WordPerSec = 149,37
Update = 13100, Epoch = 9, LR = 0,000524, AvgCost = 218,3730, Sent = 166, SentPerMin = 181,66, WordPerSec = 167,40
Update = 13200, Epoch = 9, LR = 0,000522, AvgCost = 218,3480, Sent = 463, SentPerMin = 180,76, WordPerSec = 166,85
Update = 13300, Epoch = 9, LR = 0,000520, AvgCost = 216,0002, Sent = 761, SentPerMin = 181,73, WordPerSec = 167,35
Update = 13380, Epoch = 9, LR = 0,000519, AvgCost = 214,1952, Sent = 1000, SentPerMin = 184,35, WordPerSec = 169,19
Update = 13400, Epoch = 10, LR = 0,000518, AvgCost = 210,0923, Sent = 64, SentPerMin = 166,00, WordPerSec = 144,73
Update = 13500, Epoch = 10, LR = 0,000516, AvgCost = 218,0951, Sent = 352, SentPerMin = 160,16, WordPerSec = 147,29
Update = 13600, Epoch = 10, LR = 0,000514, AvgCost = 215,1747, Sent = 645, SentPerMin = 161,05, WordPerSec = 148,24
Update = 13700, Epoch = 10, LR = 0,000513, AvgCost = 214,2235, Sent = 948, SentPerMin = 161,88, WordPerSec = 148,82
Update = 13718, Epoch = 10, LR = 0,000512, AvgCost = 213,8100, Sent = 1000, SentPerMin = 162,59, WordPerSec = 149,22
Update = 13800, Epoch = 11, LR = 0,000511, AvgCost = 214,8103, Sent = 239, SentPerMin = 186,25, WordPerSec = 170,09
Update = 13900, Epoch = 11, LR = 0,000509, AvgCost = 215,0908, Sent = 536, SentPerMin = 182,11, WordPerSec = 167,40
Update = 14000, Epoch = 11, LR = 0,000507, AvgCost = 213,2948, Sent = 831, SentPerMin = 183,43, WordPerSec = 168,43
Update = 14056, Epoch = 11, LR = 0,000506, AvgCost = 212,7429, Sent = 1000, SentPerMin = 184,42, WordPerSec = 169,25
Update = 14100, Epoch = 12, LR = 0,000505, AvgCost = 211,3024, Sent = 131, SentPerMin = 165,06, WordPerSec = 146,89
Update = 14200, Epoch = 12, LR = 0,000504, AvgCost = 215,2284, Sent = 423, SentPerMin = 160,12, WordPerSec = 147,15
Update = 14300, Epoch = 12, LR = 0,000502, AvgCost = 212,2105, Sent = 720, SentPerMin = 162,42, WordPerSec = 148,64
Update = 14394, Epoch = 12, LR = 0,000500, AvgCost = 212,4300, Sent = 1000, SentPerMin = 162,47, WordPerSec = 149,11
Update = 14400, Epoch = 13, LR = 0,000500, AvgCost = 212,0873, Sent = 16, SentPerMin = 180,48, WordPerSec = 176,15
Update = 14500, Epoch = 13, LR = 0,000498, AvgCost = 219,9832, Sent = 304, SentPerMin = 181,41, WordPerSec = 169,43
Update = 14600, Epoch = 13, LR = 0,000497, AvgCost = 213,3644, Sent = 606, SentPerMin = 179,47, WordPerSec = 164,95
Update = 14700, Epoch = 13, LR = 0,000495, AvgCost = 212,6594, Sent = 902, SentPerMin = 179,72, WordPerSec = 165,52
Update = 14732, Epoch = 13, LR = 0,000494, AvgCost = 211,5108, Sent = 1000, SentPerMin = 181,22, WordPerSec = 166,32
Update = 14800, Epoch = 14, LR = 0,000493, AvgCost = 216,8778, Sent = 195, SentPerMin = 162,52, WordPerSec = 149,38
Update = 14900, Epoch = 14, LR = 0,000492, AvgCost = 212,5759, Sent = 495, SentPerMin = 161,25, WordPerSec = 147,37
Update = 15000, Epoch = 14, LR = 0,000490, AvgCost = 213,0746, Sent = 789, SentPerMin = 159,36, WordPerSec = 146,94
Update = 15070, Epoch = 14, LR = 0,000489, AvgCost = 210,9158, Sent = 1000, SentPerMin = 161,53, WordPerSec = 148,25
Update = 15100, Epoch = 15, LR = 0,000488, AvgCost = 208,0986, Sent = 92, SentPerMin = 180,71, WordPerSec = 160,38
Update = 15200, Epoch = 15, LR = 0,000487, AvgCost = 213,7029, Sent = 381, SentPerMin = 178,33, WordPerSec = 164,10
Update = 15300, Epoch = 15, LR = 0,000485, AvgCost = 210,9183, Sent = 676, SentPerMin = 180,89, WordPerSec = 166,07
Update = 15400, Epoch = 15, LR = 0,000483, AvgCost = 208,8086, Sent = 979, SentPerMin = 183,35, WordPerSec = 167,88
Update = 15408, Epoch = 15, LR = 0,000483, AvgCost = 209,8761, Sent = 1000, SentPerMin = 183,14, WordPerSec = 168,08
Update = 15500, Epoch = 16, LR = 0,000482, AvgCost = 217,0432, Sent = 265, SentPerMin = 160,43, WordPerSec = 148,89
Update = 15600, Epoch = 16, LR = 0,000480, AvgCost = 211,5793, Sent = 566, SentPerMin = 158,85, WordPerSec = 146,20
Update = 15700, Epoch = 16, LR = 0,000479, AvgCost = 210,4794, Sent = 859, SentPerMin = 159,59, WordPerSec = 147,06
Update = 15746, Epoch = 16, LR = 0,000478, AvgCost = 209,2710, Sent = 1000, SentPerMin = 161,30, WordPerSec = 148,03
Update = 15800, Epoch = 17, LR = 0,000477, AvgCost = 211,2656, Sent = 157, SentPerMin = 179,96, WordPerSec = 163,70
Update = 15900, Epoch = 17, LR = 0,000476, AvgCost = 213,3044, Sent = 450, SentPerMin = 178,06, WordPerSec = 164,19
Update = 16000, Epoch = 17, LR = 0,000474, AvgCost = 208,4555, Sent = 752, SentPerMin = 178,04, WordPerSec = 163,23
Update = 16084, Epoch = 17, LR = 0,000473, AvgCost = 208,4750, Sent = 1000, SentPerMin = 179,58, WordPerSec = 164,81
Update = 16100, Epoch = 18, LR = 0,000473, AvgCost = 201,8341, Sent = 51, SentPerMin = 163,59, WordPerSec = 141,03
Update = 16200, Epoch = 18, LR = 0,000471, AvgCost = 214,4265, Sent = 336, SentPerMin = 155,69, WordPerSec = 144,24
Update = 16300, Epoch = 18, LR = 0,000470, AvgCost = 208,9500, Sent = 636, SentPerMin = 157,59, WordPerSec = 144,39
Update = 16400, Epoch = 18, LR = 0,000469, AvgCost = 208,7360, Sent = 935, SentPerMin = 159,58, WordPerSec = 146,69
Update = 16422, Epoch = 18, LR = 0,000468, AvgCost = 208,2791, Sent = 1000, SentPerMin = 160,73, WordPerSec = 147,52
Update = 16500, Epoch = 19, LR = 0,000467, AvgCost = 210,4325, Sent = 225, SentPerMin = 186,62, WordPerSec = 171,55
Update = 16600, Epoch = 19, LR = 0,000466, AvgCost = 210,3063, Sent = 524, SentPerMin = 183,06, WordPerSec = 168,44
Update = 16700, Epoch = 19, LR = 0,000464, AvgCost = 208,5270, Sent = 819, SentPerMin = 184,28, WordPerSec = 169,50
Update = 16760, Epoch = 19, LR = 0,000463, AvgCost = 207,3686, Sent = 1000, SentPerMin = 185,72, WordPerSec = 170,45
Update = 16800, Epoch = 20, LR = 0,000463, AvgCost = 203,3482, Sent = 122, SentPerMin = 167,93, WordPerSec = 146,57
Update = 16900, Epoch = 20, LR = 0,000462, AvgCost = 210,2232, Sent = 411, SentPerMin = 161,02, WordPerSec = 148,30
Update = 17000, Epoch = 20, LR = 0,000460, AvgCost = 206,4981, Sent = 709, SentPerMin = 164,54, WordPerSec = 150,30
Update = 17098, Epoch = 20, LR = 0,000459, AvgCost = 207,1291, Sent = 1000, SentPerMin = 164,16, WordPerSec = 150,66
Update = 17100, Epoch = 21, LR = 0,000459, AvgCost = 182,9830, Sent = 6, SentPerMin = 199,87, WordPerSec = 181,55
Update = 17200, Epoch = 21, LR = 0,000457, AvgCost = 215,0255, Sent = 292, SentPerMin = 183,37, WordPerSec = 171,56
Update = 17300, Epoch = 21, LR = 0,000456, AvgCost = 208,5291, Sent = 595, SentPerMin = 183,76, WordPerSec = 169,13
Update = 17400, Epoch = 21, LR = 0,000455, AvgCost = 207,6575, Sent = 890, SentPerMin = 185,03, WordPerSec = 170,57
Update = 17436, Epoch = 21, LR = 0,000454, AvgCost = 206,3277, Sent = 1000, SentPerMin = 186,29, WordPerSec = 170,97
Update = 17500, Epoch = 22, LR = 0,000454, AvgCost = 212,6979, Sent = 184, SentPerMin = 163,73, WordPerSec = 150,42
Update = 17600, Epoch = 22, LR = 0,000452, AvgCost = 209,1479, Sent = 482, SentPerMin = 162,41, WordPerSec = 149,31
Update = 17700, Epoch = 22, LR = 0,000451, AvgCost = 209,1403, Sent = 775, SentPerMin = 161,58, WordPerSec = 149,71
Update = 17774, Epoch = 22, LR = 0,000450, AvgCost = 206,1529, Sent = 1000, SentPerMin = 164,18, WordPerSec = 150,68
Update = 17800, Epoch = 23, LR = 0,000450, AvgCost = 200,4134, Sent = 81, SentPerMin = 183,79, WordPerSec = 163,14
Update = 17900, Epoch = 23, LR = 0,000448, AvgCost = 210,0903, Sent = 368, SentPerMin = 181,35, WordPerSec = 167,83
Update = 18000, Epoch = 23, LR = 0,000447, AvgCost = 206,3600, Sent = 663, SentPerMin = 184,67, WordPerSec = 169,67
Update = 18100, Epoch = 23, LR = 0,000446, AvgCost = 204,3878, Sent = 969, SentPerMin = 186,01, WordPerSec = 170,09
Update = 18112, Epoch = 23, LR = 0,000446, AvgCost = 205,3043, Sent = 1000, SentPerMin = 185,90, WordPerSec = 170,62
Update = 18200, Epoch = 24, LR = 0,000445, AvgCost = 214,4989, Sent = 251, SentPerMin = 162,43, WordPerSec = 151,74
Update = 18300, Epoch = 24, LR = 0,000444, AvgCost = 205,9354, Sent = 556, SentPerMin = 163,44, WordPerSec = 149,60
Update = 18400, Epoch = 24, LR = 0,000442, AvgCost = 206,4757, Sent = 847, SentPerMin = 163,07, WordPerSec = 150,31
Update = 18450, Epoch = 24, LR = 0,000442, AvgCost = 205,1572, Sent = 1000, SentPerMin = 164,30, WordPerSec = 150,79
Update = 18500, Epoch = 25, LR = 0,000441, AvgCost = 206,6354, Sent = 145, SentPerMin = 184,13, WordPerSec = 167,48
Update = 18600, Epoch = 25, LR = 0,000440, AvgCost = 209,6954, Sent = 437, SentPerMin = 182,80, WordPerSec = 169,40
Update = 18700, Epoch = 25, LR = 0,000439, AvgCost = 204,2032, Sent = 739, SentPerMin = 184,63, WordPerSec = 169,25
Update = 18788, Epoch = 25, LR = 0,000438, AvgCost = 204,2560, Sent = 1000, SentPerMin = 186,22, WordPerSec = 170,90
Update = 18800, Epoch = 26, LR = 0,000438, AvgCost = 165,2750, Sent = 41, SentPerMin = 180,83, WordPerSec = 139,23
Update = 18900, Epoch = 26, LR = 0,000436, AvgCost = 209,9242, Sent = 324, SentPerMin = 161,94, WordPerSec = 150,42
Update = 19000, Epoch = 26, LR = 0,000435, AvgCost = 204,8619, Sent = 624, SentPerMin = 162,99, WordPerSec = 149,42
Update = 19100, Epoch = 26, LR = 0,000434, AvgCost = 204,5947, Sent = 921, SentPerMin = 163,76, WordPerSec = 150,67
Update = 19126, Epoch = 26, LR = 0,000434, AvgCost = 203,8258, Sent = 1000, SentPerMin = 164,36, WordPerSec = 150,85
Update = 19200, Epoch = 27, LR = 0,000433, AvgCost = 207,9080, Sent = 213, SentPerMin = 185,47, WordPerSec = 171,10
Update = 19300, Epoch = 27, LR = 0,000432, AvgCost = 206,3908, Sent = 511, SentPerMin = 182,18, WordPerSec = 167,99
Update = 19400, Epoch = 27, LR = 0,000431, AvgCost = 204,8170, Sent = 808, SentPerMin = 184,38, WordPerSec = 169,84
Update = 19464, Epoch = 27, LR = 0,000430, AvgCost = 203,0870, Sent = 1000, SentPerMin = 185,94, WordPerSec = 170,65
Update = 19500, Epoch = 28, LR = 0,000430, AvgCost = 191,5157, Sent = 113, SentPerMin = 169,36, WordPerSec = 145,08
Update = 19600, Epoch = 28, LR = 0,000429, AvgCost = 205,9182, Sent = 399, SentPerMin = 160,94, WordPerSec = 148,14
Update = 19700, Epoch = 28, LR = 0,000427, AvgCost = 203,4845, Sent = 696, SentPerMin = 163,36, WordPerSec = 149,67
Update = 19800, Epoch = 28, LR = 0,000426, AvgCost = 203,0327, Sent = 995, SentPerMin = 163,95, WordPerSec = 150,37
Update = 19802, Epoch = 28, LR = 0,000426, AvgCost = 202,9904, Sent = 1000, SentPerMin = 163,91, WordPerSec = 150,43
Update = 19900, Epoch = 29, LR = 0,000425, AvgCost = 212,0207, Sent = 280, SentPerMin = 182,19, WordPerSec = 170,95
Update = 20000, Epoch = 29, LR = 0,000424, AvgCost = 203,2321, Sent = 588, SentPerMin = 183,81, WordPerSec = 168,14
Update = 20100, Epoch = 29, LR = 0,000423, AvgCost = 203,6854, Sent = 877, SentPerMin = 121,09, WordPerSec = 111,68
Update = 20140, Epoch = 29, LR = 0,000423, AvgCost = 202,4057, Sent = 1000, SentPerMin = 127,16, WordPerSec = 116,71
Update = 20200, Epoch = 30, LR = 0,000422, AvgCost = 203,9831, Sent = 173, SentPerMin = 161,19, WordPerSec = 148,30
Update = 20300, Epoch = 30, LR = 0,000421, AvgCost = 205,6592, Sent = 469, SentPerMin = 160,78, WordPerSec = 148,30
Update = 20400, Epoch = 30, LR = 0,000420, AvgCost = 204,1150, Sent = 766, SentPerMin = 161,09, WordPerSec = 148,64
Update = 20478, Epoch = 30, LR = 0,000419, AvgCost = 201,9195, Sent = 1000, SentPerMin = 163,48, WordPerSec = 150,04
Update = 20500, Epoch = 31, LR = 0,000419, AvgCost = 192,6872, Sent = 71, SentPerMin = 189,11, WordPerSec = 163,67
Update = 20600, Epoch = 31, LR = 0,000418, AvgCost = 204,6829, Sent = 358, SentPerMin = 182,48, WordPerSec = 167,62
Update = 20700, Epoch = 31, LR = 0,000417, AvgCost = 202,3304, Sent = 652, SentPerMin = 184,18, WordPerSec = 169,13
Update = 20800, Epoch = 31, LR = 0,000416, AvgCost = 201,2309, Sent = 955, SentPerMin = 184,93, WordPerSec = 169,66
Update = 20816, Epoch = 31, LR = 0,000416, AvgCost = 201,1944, Sent = 1000, SentPerMin = 185,46, WordPerSec = 170,21
Update = 20900, Epoch = 32, LR = 0,000415, AvgCost = 202,8522, Sent = 244, SentPerMin = 165,56, WordPerSec = 151,25
Update = 21000, Epoch = 32, LR = 0,000414, AvgCost = 202,9821, Sent = 542, SentPerMin = 161,91, WordPerSec = 148,71
Update = 21100, Epoch = 32, LR = 0,000413, AvgCost = 202,2708, Sent = 835, SentPerMin = 162,61, WordPerSec = 149,60
Update = 21154, Epoch = 32, LR = 0,000413, AvgCost = 201,2419, Sent = 1000, SentPerMin = 163,02, WordPerSec = 149,62
Update = 21200, Epoch = 33, LR = 0,000412, AvgCost = 197,4167, Sent = 136, SentPerMin = 185,95, WordPerSec = 165,46
Update = 21300, Epoch = 33, LR = 0,000411, AvgCost = 203,2032, Sent = 428, SentPerMin = 180,17, WordPerSec = 165,72
Update = 21400, Epoch = 33, LR = 0,000410, AvgCost = 200,5591, Sent = 725, SentPerMin = 182,45, WordPerSec = 167,25
Update = 21492, Epoch = 33, LR = 0,000409, AvgCost = 200,5350, Sent = 1000, SentPerMin = 182,97, WordPerSec = 167,92
Update = 21500, Epoch = 34, LR = 0,000409, AvgCost = 181,4580, Sent = 23, SentPerMin = 162,49, WordPerSec = 141,65
Update = 21600, Epoch = 34, LR = 0,000408, AvgCost = 206,7435, Sent = 311, SentPerMin = 159,95, WordPerSec = 148,72
Update = 21700, Epoch = 34, LR = 0,000407, AvgCost = 202,1819, Sent = 611, SentPerMin = 159,49, WordPerSec = 146,90
Update = 21800, Epoch = 34, LR = 0,000406, AvgCost = 201,6450, Sent = 906, SentPerMin = 160,63, WordPerSec = 148,18
Update = 21830, Epoch = 34, LR = 0,000406, AvgCost = 200,3349, Sent = 1000, SentPerMin = 161,64, WordPerSec = 148,34
Update = 21900, Epoch = 35, LR = 0,000405, AvgCost = 204,0145, Sent = 200, SentPerMin = 182,80, WordPerSec = 167,75
Update = 22000, Epoch = 35, LR = 0,000405, AvgCost = 201,2151, Sent = 501, SentPerMin = 181,35, WordPerSec = 166,14
Update = 22100, Epoch = 35, LR = 0,000404, AvgCost = 201,5060, Sent = 795, SentPerMin = 181,56, WordPerSec = 167,29
Update = 22168, Epoch = 35, LR = 0,000403, AvgCost = 199,7115, Sent = 1000, SentPerMin = 183,47, WordPerSec = 168,38
Update = 22200, Epoch = 36, LR = 0,000403, AvgCost = 193,8046, Sent = 99, SentPerMin = 163,90, WordPerSec = 142,82
Update = 22300, Epoch = 36, LR = 0,000402, AvgCost = 201,7332, Sent = 388, SentPerMin = 160,06, WordPerSec = 146,70
Update = 22400, Epoch = 36, LR = 0,000401, AvgCost = 200,1497, Sent = 683, SentPerMin = 161,54, WordPerSec = 148,02
Update = 22500, Epoch = 36, LR = 0,000400, AvgCost = 199,8984, Sent = 984, SentPerMin = 161,72, WordPerSec = 148,41
Update = 22506, Epoch = 36, LR = 0,000400, AvgCost = 199,7989, Sent = 1000, SentPerMin = 161,83, WordPerSec = 148,52
Update = 22600, Epoch = 37, LR = 0,000399, AvgCost = 207,2399, Sent = 269, SentPerMin = 181,66, WordPerSec = 169,80
Update = 22700, Epoch = 37, LR = 0,000398, AvgCost = 200,4011, Sent = 573, SentPerMin = 181,63, WordPerSec = 166,82
Update = 22800, Epoch = 37, LR = 0,000397, AvgCost = 199,8215, Sent = 865, SentPerMin = 182,04, WordPerSec = 167,67
Update = 22844, Epoch = 37, LR = 0,000397, AvgCost = 198,9929, Sent = 1000, SentPerMin = 183,42, WordPerSec = 168,34
Update = 22900, Epoch = 38, LR = 0,000396, AvgCost = 202,6263, Sent = 161, SentPerMin = 159,95, WordPerSec = 146,72
Update = 23000, Epoch = 38, LR = 0,000396, AvgCost = 203,7242, Sent = 456, SentPerMin = 158,76, WordPerSec = 146,69
Update = 23100, Epoch = 38, LR = 0,000395, AvgCost = 200,1412, Sent = 757, SentPerMin = 159,46, WordPerSec = 146,67
Update = 23182, Epoch = 38, LR = 0,000394, AvgCost = 199,0051, Sent = 1000, SentPerMin = 161,56, WordPerSec = 148,27
Update = 23200, Epoch = 39, LR = 0,000394, AvgCost = 186,5168, Sent = 59, SentPerMin = 197,81, WordPerSec = 165,29
Update = 23300, Epoch = 39, LR = 0,000393, AvgCost = 203,3754, Sent = 343, SentPerMin = 180,63, WordPerSec = 167,01
Update = 23400, Epoch = 39, LR = 0,000392, AvgCost = 199,5318, Sent = 640, SentPerMin = 181,72, WordPerSec = 166,99
Update = 23500, Epoch = 39, LR = 0,000391, AvgCost = 198,3380, Sent = 943, SentPerMin = 182,45, WordPerSec = 167,47
Update = 23520, Epoch = 39, LR = 0,000391, AvgCost = 198,2817, Sent = 1000, SentPerMin = 183,31, WordPerSec = 168,23
Starting inference...
Inference results:
和在。
在在的在。
在在的在的和和和和在在在在的和和和和在在在在的在。
在在的和在在 的在的和和和和和
"


             */

            /*
             Update = 10100, Epoch = 0, LR = 0,000597, AvgCost = 267,0951, Sent = 285, SentPerMin = 908,84, WordPerSec = 853,40
Update = 10200, Epoch = 0, LR = 0,000594, AvgCost = 247,9685, Sent = 592, SentPerMin = 965,05, WordPerSec = 885,25
Update = 10300, Epoch = 0, LR = 0,000591, AvgCost = 244,9163, Sent = 882, SentPerMin = 975,78, WordPerSec = 901,45
Update = 10338, Epoch = 0, LR = 0,000590, AvgCost = 241,7704, Sent = 1000, SentPerMin = 986,59, WordPerSec = 905,46
Update = 10400, Epoch = 1, LR = 0,000588, AvgCost = 235,6220, Sent = 178, SentPerMin = 1105,34, WordPerSec = 1019,95
Update = 10500, Epoch = 1, LR = 0,000586, AvgCost = 234,4430, Sent = 475, SentPerMin = 1120,77, WordPerSec = 1032,37
Update = 10600, Epoch = 1, LR = 0,000583, AvgCost = 232,4458, Sent = 771, SentPerMin = 1117,12, WordPerSec = 1032,55
Update = 10676, Epoch = 1, LR = 0,000581, AvgCost = 229,1314, Sent = 1000, SentPerMin = 1132,61, WordPerSec = 1039,47
Update = 10700, Epoch = 2, LR = 0,000580, AvgCost = 221,7131, Sent = 76, SentPerMin = 1149,67, WordPerSec = 1014,03
Update = 10800, Epoch = 2, LR = 0,000577, AvgCost = 230,4796, Sent = 363, SentPerMin = 1121,30, WordPerSec = 1033,98
Update = 10900, Epoch = 2, LR = 0,000575, AvgCost = 227,1094, Sent = 656, SentPerMin = 1128,72, WordPerSec = 1039,24
Update = 11000, Epoch = 2, LR = 0,000572, AvgCost = 224,6093, Sent = 961, SentPerMin = 1139,89, WordPerSec = 1044,78
Update = 11014, Epoch = 2, LR = 0,000572, AvgCost = 225,0036, Sent = 1000, SentPerMin = 1139,47, WordPerSec = 1045,76
Update = 11100, Epoch = 3, LR = 0,000569, AvgCost = 227,8982, Sent = 248, SentPerMin = 1120,81, WordPerSec = 1035,40
Update = 11200, Epoch = 3, LR = 0,000567, AvgCost = 223,8815, Sent = 549, SentPerMin = 1129,50, WordPerSec = 1034,89
Update = 11300, Epoch = 3, LR = 0,000564, AvgCost = 223,8550, Sent = 841, SentPerMin = 1126,23, WordPerSec = 1037,08
Update = 11352, Epoch = 3, LR = 0,000563, AvgCost = 222,4870, Sent = 1000, SentPerMin = 1131,76, WordPerSec = 1038,69
Update = 11400, Epoch = 4, LR = 0,000562, AvgCost = 220,7606, Sent = 140, SentPerMin = 1123,66, WordPerSec = 1008,62
Update = 11500, Epoch = 4, LR = 0,000560, AvgCost = 227,5631, Sent = 431, SentPerMin = 1099,74, WordPerSec = 1019,41
Update = 11600, Epoch = 4, LR = 0,000557, AvgCost = 220,5049, Sent = 732, SentPerMin = 1125,00, WordPerSec = 1028,53
Update = 11690, Epoch = 4, LR = 0,000555, AvgCost = 220,8360, Sent = 1000, SentPerMin = 1125,46, WordPerSec = 1032,91
Update = 11700, Epoch = 5, LR = 0,000555, AvgCost = 182,1945, Sent = 33, SentPerMin = 1220,26, WordPerSec = 942,92
Update = 11800, Epoch = 5, LR = 0,000552, AvgCost = 226,7968, Sent = 317, SentPerMin = 1086,17, WordPerSec = 1011,48
Update = 11900, Epoch = 5, LR = 0,000550, AvgCost = 220,4959, Sent = 618, SentPerMin = 1109,69, WordPerSec = 1018,44
Update = 12000, Epoch = 5, LR = 0,000548, AvgCost = 220,0981, Sent = 914, SentPerMin = 1112,81, WordPerSec = 1025,45
Update = 12028, Epoch = 5, LR = 0,000547, AvgCost = 218,9767, Sent = 1000, SentPerMin = 1117,69, WordPerSec = 1025,77
Update = 12100, Epoch = 6, LR = 0,000545, AvgCost = 226,1443, Sent = 205, SentPerMin = 1077,31, WordPerSec = 1005,84
Update = 12200, Epoch = 6, LR = 0,000543, AvgCost = 219,4303, Sent = 506, SentPerMin = 1104,67, WordPerSec = 1013,24
Update = 12300, Epoch = 6, LR = 0,000541, AvgCost = 219,4390, Sent = 802, SentPerMin = 1110,75, WordPerSec = 1022,59
Update = 12366, Epoch = 6, LR = 0,000540, AvgCost = 217,6097, Sent = 1000, SentPerMin = 1121,01, WordPerSec = 1028,83
Update = 12400, Epoch = 7, LR = 0,000539, AvgCost = 207,4814, Sent = 106, SentPerMin = 1162,19, WordPerSec = 1001,02
Update = 12500, Epoch = 7, LR = 0,000537, AvgCost = 220,3300, Sent = 393, SentPerMin = 1101,86, WordPerSec = 1015,51
Update = 12600, Epoch = 7, LR = 0,000535, AvgCost = 216,8949, Sent = 689, SentPerMin = 1118,72, WordPerSec = 1025,49
Update = 12700, Epoch = 7, LR = 0,000532, AvgCost = 216,6560, Sent = 989, SentPerMin = 1121,48, WordPerSec = 1029,54
Update = 12704, Epoch = 7, LR = 0,000532, AvgCost = 216,4970, Sent = 1000, SentPerMin = 1121,58, WordPerSec = 1029,35
Update = 12800, Epoch = 8, LR = 0,000530, AvgCost = 226,3290, Sent = 275, SentPerMin = 1092,14, WordPerSec = 1025,22
Update = 12900, Epoch = 8, LR = 0,000528, AvgCost = 216,7690, Sent = 581, SentPerMin = 1116,03, WordPerSec = 1023,79
Update = 13000, Epoch = 8, LR = 0,000526, AvgCost = 217,0705, Sent = 870, SentPerMin = 1114,93, WordPerSec = 1029,39
Update = 13042, Epoch = 8, LR = 0,000525, AvgCost = 215,5757, Sent = 1000, SentPerMin = 1123,46, WordPerSec = 1031,07
Update = 13100, Epoch = 9, LR = 0,000524, AvgCost = 217,9976, Sent = 166, SentPerMin = 1092,71, WordPerSec = 1006,92
Update = 13200, Epoch = 9, LR = 0,000522, AvgCost = 218,9459, Sent = 463, SentPerMin = 1106,59, WordPerSec = 1021,38
Update = 13300, Epoch = 9, LR = 0,000520, AvgCost = 216,7678, Sent = 761, SentPerMin = 1110,25, WordPerSec = 1022,37
Update = 13380, Epoch = 9, LR = 0,000519, AvgCost = 214,7979, Sent = 1000, SentPerMin = 1122,01, WordPerSec = 1029,75
Update = 13400, Epoch = 10, LR = 0,000518, AvgCost = 207,1205, Sent = 64, SentPerMin = 1180,71, WordPerSec = 1029,43
Update = 13500, Epoch = 10, LR = 0,000516, AvgCost = 217,7570, Sent = 352, SentPerMin = 1109,51, WordPerSec = 1020,31
Update = 13600, Epoch = 10, LR = 0,000514, AvgCost = 215,1332, Sent = 645, SentPerMin = 1111,98, WordPerSec = 1023,54
Update = 13700, Epoch = 10, LR = 0,000513, AvgCost = 214,3464, Sent = 948, SentPerMin = 1120,93, WordPerSec = 1030,48
Update = 13718, Epoch = 10, LR = 0,000512, AvgCost = 213,9009, Sent = 1000, SentPerMin = 1123,62, WordPerSec = 1031,22
Update = 13800, Epoch = 11, LR = 0,000511, AvgCost = 215,0094, Sent = 239, SentPerMin = 1122,87, WordPerSec = 1025,46
Update = 13900, Epoch = 11, LR = 0,000509, AvgCost = 215,2400, Sent = 536, SentPerMin = 1115,06, WordPerSec = 1024,95
Update = 14000, Epoch = 11, LR = 0,000507, AvgCost = 213,9423, Sent = 831, SentPerMin = 1115,18, WordPerSec = 1024,00
Update = 14056, Epoch = 11, LR = 0,000506, AvgCost = 213,3222, Sent = 1000, SentPerMin = 1119,85, WordPerSec = 1027,76
Update = 14100, Epoch = 12, LR = 0,000505, AvgCost = 212,8110, Sent = 131, SentPerMin = 1011,55, WordPerSec = 900,22
Update = 14200, Epoch = 12, LR = 0,000504, AvgCost = 215,4806, Sent = 423, SentPerMin = 984,54, WordPerSec = 904,78
Update = 14300, Epoch = 12, LR = 0,000502, AvgCost = 212,5254, Sent = 720, SentPerMin = 998,58, WordPerSec = 913,86
Update = 14394, Epoch = 12, LR = 0,000500, AvgCost = 212,7428, Sent = 1000, SentPerMin = 1000,29, WordPerSec = 918,04
Update = 14400, Epoch = 13, LR = 0,000500, AvgCost = 215,3591, Sent = 16, SentPerMin = 1034,03, WordPerSec = 1009,25
Update = 14500, Epoch = 13, LR = 0,000498, AvgCost = 220,6721, Sent = 304, SentPerMin = 1098,46, WordPerSec = 1025,95
Update = 14600, Epoch = 13, LR = 0,000497, AvgCost = 213,4914, Sent = 606, SentPerMin = 1114,03, WordPerSec = 1023,86
Update = 14700, Epoch = 13, LR = 0,000495, AvgCost = 213,1514, Sent = 902, SentPerMin = 1116,93, WordPerSec = 1028,68
Update = 14732, Epoch = 13, LR = 0,000494, AvgCost = 211,9151, Sent = 1000, SentPerMin = 1123,64, WordPerSec = 1031,24
Update = 14800, Epoch = 14, LR = 0,000493, AvgCost = 217,2367, Sent = 195, SentPerMin = 1105,96, WordPerSec = 1016,54
Update = 14900, Epoch = 14, LR = 0,000492, AvgCost = 212,7880, Sent = 495, SentPerMin = 1118,99, WordPerSec = 1022,65
Update = 15000, Epoch = 14, LR = 0,000490, AvgCost = 213,5640, Sent = 789, SentPerMin = 1113,56, WordPerSec = 1026,79
Update = 15070, Epoch = 14, LR = 0,000489, AvgCost = 211,2479, Sent = 1000, SentPerMin = 1125,11, WordPerSec = 1032,59
Update = 15100, Epoch = 15, LR = 0,000488, AvgCost = 208,6926, Sent = 92, SentPerMin = 1131,20, WordPerSec = 1003,94
Update = 15200, Epoch = 15, LR = 0,000487, AvgCost = 214,0040, Sent = 381, SentPerMin = 1105,92, WordPerSec = 1017,68
Update = 15300, Epoch = 15, LR = 0,000485, AvgCost = 211,5242, Sent = 676, SentPerMin = 1113,84, WordPerSec = 1022,58
Update = 15400, Epoch = 15, LR = 0,000483, AvgCost = 209,3584, Sent = 979, SentPerMin = 1120,96, WordPerSec = 1026,42
Update = 15408, Epoch = 15, LR = 0,000483, AvgCost = 210,4006, Sent = 1000, SentPerMin = 1118,53, WordPerSec = 1026,55
Update = 15500, Epoch = 16, LR = 0,000482, AvgCost = 217,2059, Sent = 265, SentPerMin = 1097,25, WordPerSec = 1018,30
Update = 15600, Epoch = 16, LR = 0,000480, AvgCost = 211,8686, Sent = 566, SentPerMin = 1105,54, WordPerSec = 1017,51
Update = 15700, Epoch = 16, LR = 0,000479, AvgCost = 211,2000, Sent = 859, SentPerMin = 1105,71, WordPerSec = 1018,91
Update = 15746, Epoch = 16, LR = 0,000478, AvgCost = 209,7898, Sent = 1000, SentPerMin = 1115,54, WordPerSec = 1023,81
Update = 15800, Epoch = 17, LR = 0,000477, AvgCost = 211,6762, Sent = 157, SentPerMin = 1097,60, WordPerSec = 998,44
Update = 15900, Epoch = 17, LR = 0,000476, AvgCost = 213,7504, Sent = 450, SentPerMin = 1098,53, WordPerSec = 1012,96
Update = 16000, Epoch = 17, LR = 0,000474, AvgCost = 209,2934, Sent = 752, SentPerMin = 1103,25, WordPerSec = 1011,48
Update = 16084, Epoch = 17, LR = 0,000473, AvgCost = 209,1424, Sent = 1000, SentPerMin = 1111,46, WordPerSec = 1020,06
Update = 16100, Epoch = 18, LR = 0,000473, AvgCost = 202,6533, Sent = 51, SentPerMin = 1170,82, WordPerSec = 1009,36
Update = 16200, Epoch = 18, LR = 0,000471, AvgCost = 214,6366, Sent = 336, SentPerMin = 1094,68, WordPerSec = 1014,21
Update = 16300, Epoch = 18, LR = 0,000470, AvgCost = 209,1314, Sent = 636, SentPerMin = 1108,60, WordPerSec = 1015,73
Update = 16400, Epoch = 18, LR = 0,000469, AvgCost = 208,9439, Sent = 935, SentPerMin = 1114,57, WordPerSec = 1024,57
Update = 16422, Epoch = 18, LR = 0,000468, AvgCost = 208,4664, Sent = 1000, SentPerMin = 1119,40, WordPerSec = 1027,35
Update = 16500, Epoch = 19, LR = 0,000467, AvgCost = 211,6079, Sent = 225, SentPerMin = 1101,81, WordPerSec = 1012,85
Update = 16600, Epoch = 19, LR = 0,000466, AvgCost = 210,8981, Sent = 524, SentPerMin = 1098,48, WordPerSec = 1010,75
Update = 16700, Epoch = 19, LR = 0,000464, AvgCost = 209,2099, Sent = 819, SentPerMin = 1105,21, WordPerSec = 1016,57
Update = 16760, Epoch = 19, LR = 0,000463, AvgCost = 207,9460, Sent = 1000, SentPerMin = 1112,42, WordPerSec = 1020,94
Update = 16800, Epoch = 20, LR = 0,000463, AvgCost = 203,0082, Sent = 122, SentPerMin = 1152,75, WordPerSec = 1006,14
Update = 16900, Epoch = 20, LR = 0,000462, AvgCost = 210,6095, Sent = 411, SentPerMin = 1101,98, WordPerSec = 1014,93
Update = 17000, Epoch = 20, LR = 0,000460, AvgCost = 206,8985, Sent = 709, SentPerMin = 1122,32, WordPerSec = 1025,18
Update = 17098, Epoch = 20, LR = 0,000459, AvgCost = 207,3357, Sent = 1000, SentPerMin = 1120,09, WordPerSec = 1027,98
Update = 17100, Epoch = 21, LR = 0,000459, AvgCost = 181,9296, Sent = 6, SentPerMin = 1034,20, WordPerSec = 939,39
Update = 17200, Epoch = 21, LR = 0,000457, AvgCost = 215,5504, Sent = 292, SentPerMin = 1072,68, WordPerSec = 1003,62
Update = 17300, Epoch = 21, LR = 0,000456, AvgCost = 208,9926, Sent = 595, SentPerMin = 1097,97, WordPerSec = 1010,56
Update = 17400, Epoch = 21, LR = 0,000455, AvgCost = 208,1172, Sent = 890, SentPerMin = 1106,28, WordPerSec = 1019,79
Update = 17436, Epoch = 21, LR = 0,000454, AvgCost = 206,7244, Sent = 1000, SentPerMin = 1114,26, WordPerSec = 1022,63
Update = 17500, Epoch = 22, LR = 0,000454, AvgCost = 212,5636, Sent = 184, SentPerMin = 1099,52, WordPerSec = 1010,18
Update = 17600, Epoch = 22, LR = 0,000452, AvgCost = 209,3608, Sent = 482, SentPerMin = 1110,11, WordPerSec = 1020,59
Update = 17700, Epoch = 22, LR = 0,000451, AvgCost = 209,3534, Sent = 775, SentPerMin = 1091,79, WordPerSec = 1011,56
Update = 17774, Epoch = 22, LR = 0,000450, AvgCost = 206,1381, Sent = 1000, SentPerMin = 1101,68, WordPerSec = 1011,09
Update = 17800, Epoch = 23, LR = 0,000450, AvgCost = 200,1327, Sent = 81, SentPerMin = 1119,56, WordPerSec = 993,78
Update = 17900, Epoch = 23, LR = 0,000448, AvgCost = 210,8632, Sent = 368, SentPerMin = 1083,12, WordPerSec = 1002,37
Update = 18000, Epoch = 23, LR = 0,000447, AvgCost = 206,9487, Sent = 663, SentPerMin = 1099,45, WordPerSec = 1010,16
Update = 18100, Epoch = 23, LR = 0,000446, AvgCost = 204,6826, Sent = 969, SentPerMin = 1113,95, WordPerSec = 1018,63
Update = 18112, Epoch = 23, LR = 0,000446, AvgCost = 205,5879, Sent = 1000, SentPerMin = 1111,10, WordPerSec = 1019,73
Update = 18200, Epoch = 24, LR = 0,000445, AvgCost = 215,9936, Sent = 251, SentPerMin = 969,93, WordPerSec = 906,10
Update = 18300, Epoch = 24, LR = 0,000444, AvgCost = 206,7988, Sent = 556, SentPerMin = 993,86, WordPerSec = 909,70
Update = 18400, Epoch = 24, LR = 0,000442, AvgCost = 207,0428, Sent = 847, SentPerMin = 989,37, WordPerSec = 911,90
Update = 18450, Epoch = 24, LR = 0,000442, AvgCost = 205,7148, Sent = 1000, SentPerMin = 996,40, WordPerSec = 914,46
Update = 18500, Epoch = 25, LR = 0,000441, AvgCost = 208,0296, Sent = 145, SentPerMin = 1094,32, WordPerSec = 995,33
Update = 18600, Epoch = 25, LR = 0,000440, AvgCost = 210,9122, Sent = 437, SentPerMin = 1088,89, WordPerSec = 1009,07
Update = 18700, Epoch = 25, LR = 0,000439, AvgCost = 204,7475, Sent = 739, SentPerMin = 1106,85, WordPerSec = 1014,64
Update = 18788, Epoch = 25, LR = 0,000438, AvgCost = 204,8364, Sent = 1000, SentPerMin = 1117,80, WordPerSec = 1025,88
Update = 18800, Epoch = 26, LR = 0,000438, AvgCost = 167,0611, Sent = 41, SentPerMin = 1261,11, WordPerSec = 970,95
Update = 18900, Epoch = 26, LR = 0,000436, AvgCost = 211,3974, Sent = 324, SentPerMin = 1107,66, WordPerSec = 1028,86
Update = 19000, Epoch = 26, LR = 0,000435, AvgCost = 205,3897, Sent = 624, SentPerMin = 1124,21, WordPerSec = 1030,62
Update = 19100, Epoch = 26, LR = 0,000434, AvgCost = 204,9677, Sent = 921, SentPerMin = 1128,37, WordPerSec = 1038,15
Update = 19126, Epoch = 26, LR = 0,000434, AvgCost = 204,2264, Sent = 1000, SentPerMin = 1131,94, WordPerSec = 1038,86
Update = 19200, Epoch = 27, LR = 0,000433, AvgCost = 209,6262, Sent = 213, SentPerMin = 1101,53, WordPerSec = 1016,20
Update = 19300, Epoch = 27, LR = 0,000432, AvgCost = 207,1482, Sent = 511, SentPerMin = 1106,01, WordPerSec = 1019,87
Update = 19400, Epoch = 27, LR = 0,000431, AvgCost = 205,3298, Sent = 808, SentPerMin = 1117,30, WordPerSec = 1029,17
Update = 19464, Epoch = 27, LR = 0,000430, AvgCost = 203,6182, Sent = 1000, SentPerMin = 1126,48, WordPerSec = 1033,85
Update = 19500, Epoch = 28, LR = 0,000430, AvgCost = 192,7775, Sent = 113, SentPerMin = 1175,38, WordPerSec = 1006,88
Update = 19600, Epoch = 28, LR = 0,000429, AvgCost = 206,3538, Sent = 399, SentPerMin = 1095,14, WordPerSec = 1008,04
Update = 19700, Epoch = 28, LR = 0,000427, AvgCost = 203,4598, Sent = 696, SentPerMin = 1114,37, WordPerSec = 1020,99
Update = 19800, Epoch = 28, LR = 0,000426, AvgCost = 203,1576, Sent = 995, SentPerMin = 1119,25, WordPerSec = 1026,50
Update = 19802, Epoch = 28, LR = 0,000426, AvgCost = 203,1176, Sent = 1000, SentPerMin = 1118,47, WordPerSec = 1026,49
Update = 19900, Epoch = 29, LR = 0,000425, AvgCost = 212,8838, Sent = 280, SentPerMin = 1091,96, WordPerSec = 1024,62
Update = 20000, Epoch = 29, LR = 0,000424, AvgCost = 203,2933, Sent = 588, SentPerMin = 1116,81, WordPerSec = 1021,59
Update = 20100, Epoch = 29, LR = 0,000423, AvgCost = 203,7607, Sent = 877, SentPerMin = 786,16, WordPerSec = 725,04
Update = 20140, Epoch = 29, LR = 0,000423, AvgCost = 202,5788, Sent = 1000, SentPerMin = 820,40, WordPerSec = 752,94
Update = 20200, Epoch = 30, LR = 0,000422, AvgCost = 205,0210, Sent = 173, SentPerMin = 1095,07, WordPerSec = 1007,51
Update = 20300, Epoch = 30, LR = 0,000421, AvgCost = 206,0275, Sent = 469, SentPerMin = 1100,75, WordPerSec = 1015,36
Update = 20400, Epoch = 30, LR = 0,000420, AvgCost = 204,4658, Sent = 766, SentPerMin = 1103,41, WordPerSec = 1018,16
Update = 20478, Epoch = 30, LR = 0,000419, AvgCost = 202,2282, Sent = 1000, SentPerMin = 1117,51, WordPerSec = 1025,61
Update = 20500, Epoch = 31, LR = 0,000419, AvgCost = 194,2148, Sent = 71, SentPerMin = 1167,56, WordPerSec = 1010,52
Update = 20600, Epoch = 31, LR = 0,000418, AvgCost = 205,6829, Sent = 358, SentPerMin = 1104,30, WordPerSec = 1014,38
Update = 20700, Epoch = 31, LR = 0,000417, AvgCost = 202,9371, Sent = 652, SentPerMin = 1110,23, WordPerSec = 1019,50
Update = 20800, Epoch = 31, LR = 0,000416, AvgCost = 201,7467, Sent = 955, SentPerMin = 1117,03, WordPerSec = 1024,80
Update = 20816, Epoch = 31, LR = 0,000416, AvgCost = 201,8086, Sent = 1000, SentPerMin = 1116,92, WordPerSec = 1025,07
Update = 20900, Epoch = 32, LR = 0,000415, AvgCost = 203,9782, Sent = 244, SentPerMin = 1105,85, WordPerSec = 1010,30
Update = 21000, Epoch = 32, LR = 0,000414, AvgCost = 203,3937, Sent = 542, SentPerMin = 1103,38, WordPerSec = 1013,40
Update = 21100, Epoch = 32, LR = 0,000413, AvgCost = 202,6720, Sent = 835, SentPerMin = 1106,72, WordPerSec = 1018,21
Update = 21154, Epoch = 32, LR = 0,000413, AvgCost = 201,5295, Sent = 1000, SentPerMin = 1114,73, WordPerSec = 1023,06
Update = 21200, Epoch = 33, LR = 0,000412, AvgCost = 199,1334, Sent = 136, SentPerMin = 1130,90, WordPerSec = 1006,30
Update = 21300, Epoch = 33, LR = 0,000411, AvgCost = 203,9710, Sent = 428, SentPerMin = 1095,69, WordPerSec = 1007,80
Update = 21400, Epoch = 33, LR = 0,000410, AvgCost = 201,2110, Sent = 725, SentPerMin = 1109,07, WordPerSec = 1016,65
Update = 21492, Epoch = 33, LR = 0,000409, AvgCost = 201,0861, Sent = 1000, SentPerMin = 1113,52, WordPerSec = 1021,95
Update = 21500, Epoch = 34, LR = 0,000409, AvgCost = 182,7406, Sent = 23, SentPerMin = 1084,12, WordPerSec = 945,07
Update = 21600, Epoch = 34, LR = 0,000408, AvgCost = 207,8671, Sent = 311, SentPerMin = 1088,56, WordPerSec = 1012,08
Update = 21700, Epoch = 34, LR = 0,000407, AvgCost = 202,9047, Sent = 611, SentPerMin = 1094,81, WordPerSec = 1008,36
Update = 21800, Epoch = 34, LR = 0,000406, AvgCost = 202,1781, Sent = 906, SentPerMin = 1102,44, WordPerSec = 1017,00
Update = 21830, Epoch = 34, LR = 0,000406, AvgCost = 200,8633, Sent = 1000, SentPerMin = 1110,72, WordPerSec = 1019,39
Update = 21900, Epoch = 35, LR = 0,000405, AvgCost = 206,3314, Sent = 200, SentPerMin = 969,41, WordPerSec = 889,59
Update = 22000, Epoch = 35, LR = 0,000405, AvgCost = 203,2067, Sent = 501, SentPerMin = 981,67, WordPerSec = 899,34
Update = 22100, Epoch = 35, LR = 0,000404, AvgCost = 203,4333, Sent = 795, SentPerMin = 982,43, WordPerSec = 905,22
Update = 22168, Epoch = 35, LR = 0,000403, AvgCost = 201,4581, Sent = 1000, SentPerMin = 993,49, WordPerSec = 911,80
Update = 22200, Epoch = 36, LR = 0,000403, AvgCost = 195,9827, Sent = 99, SentPerMin = 1183,06, WordPerSec = 1030,90
Update = 22300, Epoch = 36, LR = 0,000402, AvgCost = 203,4232, Sent = 388, SentPerMin = 1142,97, WordPerSec = 1047,57
Update = 22400, Epoch = 36, LR = 0,000401, AvgCost = 201,4280, Sent = 683, SentPerMin = 1148,50, WordPerSec = 1052,38
Update = 22500, Epoch = 36, LR = 0,000400, AvgCost = 200,9216, Sent = 984, SentPerMin = 1150,66, WordPerSec = 1055,96
Update = 22506, Epoch = 36, LR = 0,000400, AvgCost = 200,8182, Sent = 1000, SentPerMin = 1150,09, WordPerSec = 1055,52
Update = 22600, Epoch = 37, LR = 0,000399, AvgCost = 208,8741, Sent = 269, SentPerMin = 1117,55, WordPerSec = 1044,57
Update = 22700, Epoch = 37, LR = 0,000398, AvgCost = 201,8677, Sent = 573, SentPerMin = 1136,85, WordPerSec = 1044,10
Update = 22800, Epoch = 37, LR = 0,000397, AvgCost = 201,2840, Sent = 865, SentPerMin = 1136,25, WordPerSec = 1046,53
Update = 22844, Epoch = 37, LR = 0,000397, AvgCost = 200,3513, Sent = 1000, SentPerMin = 1143,58, WordPerSec = 1049,54
Update = 22900, Epoch = 38, LR = 0,000396, AvgCost = 204,4266, Sent = 161, SentPerMin = 1121,53, WordPerSec = 1028,77
Update = 23000, Epoch = 38, LR = 0,000396, AvgCost = 204,5975, Sent = 456, SentPerMin = 1129,16, WordPerSec = 1043,32
Update = 23100, Epoch = 38, LR = 0,000395, AvgCost = 201,0247, Sent = 757, SentPerMin = 1138,04, WordPerSec = 1046,79
Update = 23182, Epoch = 38, LR = 0,000394, AvgCost = 199,9245, Sent = 1000, SentPerMin = 1147,58, WordPerSec = 1053,21
Update = 23200, Epoch = 39, LR = 0,000394, AvgCost = 187,7877, Sent = 59, SentPerMin = 1246,61, WordPerSec = 1041,66
Update = 23300, Epoch = 39, LR = 0,000393, AvgCost = 204,6264, Sent = 343, SentPerMin = 1128,58, WordPerSec = 1043,47
Update = 23400, Epoch = 39, LR = 0,000392, AvgCost = 200,5927, Sent = 640, SentPerMin = 1136,32, WordPerSec = 1044,20
Update = 23500, Epoch = 39, LR = 0,000391, AvgCost = 199,4495, Sent = 943, SentPerMin = 1146,27, WordPerSec = 1052,15
Update = 23520, Epoch = 39, LR = 0,000391, AvgCost = 199,4859, Sent = 1000, SentPerMin = 1147,10, WordPerSec = 1052,77
Update = 23600, Epoch = 40, LR = 0,000391, AvgCost = 203,1298, Sent = 231, SentPerMin = 1133,66, WordPerSec = 1043,20
Update = 23700, Epoch = 40, LR = 0,000390, AvgCost = 201,7103, Sent = 529, SentPerMin = 1131,29, WordPerSec = 1041,22
Update = 23800, Epoch = 40, LR = 0,000389, AvgCost = 200,3323, Sent = 824, SentPerMin = 1136,64, WordPerSec = 1045,92
Update = 23858, Epoch = 40, LR = 0,000388, AvgCost = 199,1810, Sent = 1000, SentPerMin = 1146,18, WordPerSec = 1051,93
Update = 23900, Epoch = 41, LR = 0,000388, AvgCost = 197,2178, Sent = 126, SentPerMin = 1156,83, WordPerSec = 1021,26
Update = 24000, Epoch = 41, LR = 0,000387, AvgCost = 201,3109, Sent = 418, SentPerMin = 1129,30, WordPerSec = 1036,27
Update = 24100, Epoch = 41, LR = 0,000386, AvgCost = 198,5880, Sent = 715, SentPerMin = 1144,04, WordPerSec = 1046,22
Update = 24196, Epoch = 41, LR = 0,000386, AvgCost = 198,9409, Sent = 1000, SentPerMin = 1146,37, WordPerSec = 1052,10
Update = 24200, Epoch = 42, LR = 0,000386, AvgCost = 177,0896, Sent = 12, SentPerMin = 1195,67, WordPerSec = 1076,10
Update = 24300, Epoch = 42, LR = 0,000385, AvgCost = 206,6823, Sent = 299, SentPerMin = 1127,09, WordPerSec = 1051,33
Update = 24400, Epoch = 42, LR = 0,000384, AvgCost = 200,0726, Sent = 601, SentPerMin = 1140,46, WordPerSec = 1047,67
Update = 24500, Epoch = 42, LR = 0,000383, AvgCost = 199,3256, Sent = 897, SentPerMin = 1145,64, WordPerSec = 1053,79
Update = 24534, Epoch = 42, LR = 0,000383, AvgCost = 198,5700, Sent = 1000, SentPerMin = 1150,60, WordPerSec = 1055,98
Starting inference...
Inference results:
和和和和和的的的,在 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的 的
和的和和
和和和和的的的的的的的的的的的的的的的的的。
和的和和
和的和和和

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
                EncoderLayerDepth = 1,
                DecoderLayerDepth = 1,
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
                MaxEpochNum = 46, // 43, // 40, // 26, // 21, // 13, // 3,
                SharedEmbeddings = false,
                ModelFilePath = "seq2seq_test.model",
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
