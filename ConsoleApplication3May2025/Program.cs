using Seq2SeqSharp;
using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Enums;
using Seq2SeqSharp.LearningRate;
using Seq2SeqSharp.Metrics;
using Seq2SeqSharp.Optimizer;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;

// put sentencepiece dll in exe folder
namespace ConsoleApplication3May2025
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
            options.ModelFilePath = "seq2seq_test86.model.test";

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
   Update = 10100, Epoch = 0, LR = 0,000597, AvgCost = 266,6072, Sent = 285, SentPerMin = 130,26, WordPerSec = 122,31
Update = 10200, Epoch = 0, LR = 0,000594, AvgCost = 247,7742, Sent = 592, SentPerMin = 132,49, WordPerSec = 121,53
Update = 10300, Epoch = 0, LR = 0,000591, AvgCost = 244,8227, Sent = 882, SentPerMin = 133,18, WordPerSec = 123,03
Update = 10338, Epoch = 0, LR = 0,000590, AvgCost = 241,6965, Sent = 1000, SentPerMin = 134,55, WordPerSec = 123,48
Update = 10400, Epoch = 1, LR = 0,000588, AvgCost = 233,9947, Sent = 178, SentPerMin = 165,21, WordPerSec = 152,45
Update = 10500, Epoch = 1, LR = 0,000586, AvgCost = 233,9560, Sent = 475, SentPerMin = 164,57, WordPerSec = 151,59
Update = 10600, Epoch = 1, LR = 0,000583, AvgCost = 232,3062, Sent = 771, SentPerMin = 162,32, WordPerSec = 150,04
Update = 10676, Epoch = 1, LR = 0,000581, AvgCost = 229,0814, Sent = 1000, SentPerMin = 163,80, WordPerSec = 150,33
Update = 10700, Epoch = 2, LR = 0,000580, AvgCost = 220,6357, Sent = 76, SentPerMin = 128,89, WordPerSec = 113,69
Update = 10800, Epoch = 2, LR = 0,000577, AvgCost = 230,4737, Sent = 363, SentPerMin = 128,89, WordPerSec = 118,85
Update = 10900, Epoch = 2, LR = 0,000575, AvgCost = 227,3471, Sent = 656, SentPerMin = 129,83, WordPerSec = 119,54
Update = 11000, Epoch = 2, LR = 0,000572, AvgCost = 224,8045, Sent = 961, SentPerMin = 131,09, WordPerSec = 120,15
Update = 11014, Epoch = 2, LR = 0,000572, AvgCost = 225,1980, Sent = 1000, SentPerMin = 131,32, WordPerSec = 120,52
Starting inference...
Inference results:
,,,,,,,,,,,,,,,,,,,。
,,,,,,,,,,,,,,,。
,,,,,,,,,,,,,,,,,,,,,。 。
,,,,,,,,,,,,。
,,,,,,,,,,。

             */

            /*
             Update = 10100, Epoch = 0, LR = 0,000597, AvgCost = 266,0792, Sent = 285, SentPerMin = 126,60, WordPerSec = 118,88
Update = 10200, Epoch = 0, LR = 0,000594, AvgCost = 247,4899, Sent = 592, SentPerMin = 127,78, WordPerSec = 117,22
Update = 10300, Epoch = 0, LR = 0,000591, AvgCost = 244,4198, Sent = 882, SentPerMin = 128,24, WordPerSec = 118,47
Update = 10338, Epoch = 0, LR = 0,000590, AvgCost = 241,3129, Sent = 1000, SentPerMin = 129,39, WordPerSec = 118,75
Update = 10400, Epoch = 1, LR = 0,000588, AvgCost = 232,7609, Sent = 178, SentPerMin = 157,45, WordPerSec = 145,28
Update = 10500, Epoch = 1, LR = 0,000586, AvgCost = 233,3160, Sent = 475, SentPerMin = 157,32, WordPerSec = 144,91
Update = 10600, Epoch = 1, LR = 0,000583, AvgCost = 231,7656, Sent = 771, SentPerMin = 157,26, WordPerSec = 145,35
Update = 10676, Epoch = 1, LR = 0,000581, AvgCost = 228,5725, Sent = 1000, SentPerMin = 159,77, WordPerSec = 146,63
Update = 10700, Epoch = 2, LR = 0,000580, AvgCost = 219,7719, Sent = 76, SentPerMin = 128,66, WordPerSec = 113,48
Update = 10800, Epoch = 2, LR = 0,000577, AvgCost = 230,1567, Sent = 363, SentPerMin = 127,88, WordPerSec = 117,92
Update = 10900, Epoch = 2, LR = 0,000575, AvgCost = 227,1536, Sent = 656, SentPerMin = 128,82, WordPerSec = 118,61
Update = 11000, Epoch = 2, LR = 0,000572, AvgCost = 224,5206, Sent = 961, SentPerMin = 129,76, WordPerSec = 118,94
Update = 11014, Epoch = 2, LR = 0,000572, AvgCost = 224,9015, Sent = 1000, SentPerMin = 129,97, WordPerSec = 119,28
Update = 11100, Epoch = 3, LR = 0,000569, AvgCost = 228,0308, Sent = 248, SentPerMin = 160,07, WordPerSec = 147,87
Update = 11200, Epoch = 3, LR = 0,000567, AvgCost = 223,9975, Sent = 549, SentPerMin = 158,92, WordPerSec = 145,61
Update = 11300, Epoch = 3, LR = 0,000564, AvgCost = 224,0151, Sent = 841, SentPerMin = 159,21, WordPerSec = 146,61
Update = 11352, Epoch = 3, LR = 0,000563, AvgCost = 222,5293, Sent = 1000, SentPerMin = 160,41, WordPerSec = 147,22
Update = 11400, Epoch = 4, LR = 0,000562, AvgCost = 219,4508, Sent = 140, SentPerMin = 131,66, WordPerSec = 118,18
Update = 11500, Epoch = 4, LR = 0,000560, AvgCost = 227,0221, Sent = 431, SentPerMin = 127,60, WordPerSec = 118,28
Update = 11600, Epoch = 4, LR = 0,000557, AvgCost = 220,3995, Sent = 732, SentPerMin = 129,90, WordPerSec = 118,76
Update = 11690, Epoch = 4, LR = 0,000555, AvgCost = 220,6876, Sent = 1000, SentPerMin = 129,89, WordPerSec = 119,21
Update = 11700, Epoch = 5, LR = 0,000555, AvgCost = 181,9188, Sent = 33, SentPerMin = 176,59, WordPerSec = 136,46
Update = 11800, Epoch = 5, LR = 0,000552, AvgCost = 226,6946, Sent = 317, SentPerMin = 157,25, WordPerSec = 146,43
Update = 11900, Epoch = 5, LR = 0,000550, AvgCost = 220,6362, Sent = 618, SentPerMin = 158,29, WordPerSec = 145,27
Update = 12000, Epoch = 5, LR = 0,000548, AvgCost = 220,1790, Sent = 914, SentPerMin = 159,07, WordPerSec = 146,58
Update = 12028, Epoch = 5, LR = 0,000547, AvgCost = 218,9865, Sent = 1000, SentPerMin = 160,05, WordPerSec = 146,89
Update = 12100, Epoch = 6, LR = 0,000545, AvgCost = 226,8031, Sent = 205, SentPerMin = 127,94, WordPerSec = 119,45
Update = 12200, Epoch = 6, LR = 0,000543, AvgCost = 219,6602, Sent = 506, SentPerMin = 127,91, WordPerSec = 117,32
Update = 12300, Epoch = 6, LR = 0,000541, AvgCost = 219,3804, Sent = 802, SentPerMin = 128,69, WordPerSec = 118,48
Update = 12366, Epoch = 6, LR = 0,000540, AvgCost = 217,4785, Sent = 1000, SentPerMin = 129,81, WordPerSec = 119,13
Update = 12400, Epoch = 7, LR = 0,000539, AvgCost = 206,9532, Sent = 106, SentPerMin = 162,08, WordPerSec = 139,61
Update = 12500, Epoch = 7, LR = 0,000537, AvgCost = 219,7575, Sent = 393, SentPerMin = 156,66, WordPerSec = 144,38
Update = 12600, Epoch = 7, LR = 0,000535, AvgCost = 216,3740, Sent = 689, SentPerMin = 160,07, WordPerSec = 146,73
Update = 12700, Epoch = 7, LR = 0,000532, AvgCost = 216,0075, Sent = 989, SentPerMin = 160,07, WordPerSec = 146,95
Update = 12704, Epoch = 7, LR = 0,000532, AvgCost = 215,8589, Sent = 1000, SentPerMin = 160,19, WordPerSec = 147,02
Update = 12800, Epoch = 8, LR = 0,000530, AvgCost = 226,1343, Sent = 275, SentPerMin = 127,28, WordPerSec = 119,48
Update = 12900, Epoch = 8, LR = 0,000528, AvgCost = 216,1646, Sent = 581, SentPerMin = 128,58, WordPerSec = 117,96
Update = 13000, Epoch = 8, LR = 0,000526, AvgCost = 216,4873, Sent = 870, SentPerMin = 129,64, WordPerSec = 119,70
Update = 13042, Epoch = 8, LR = 0,000525, AvgCost = 215,0440, Sent = 1000, SentPerMin = 130,41, WordPerSec = 119,69
Update = 13100, Epoch = 9, LR = 0,000524, AvgCost = 218,3432, Sent = 166, SentPerMin = 157,54, WordPerSec = 145,17
Update = 13200, Epoch = 9, LR = 0,000522, AvgCost = 218,2000, Sent = 463, SentPerMin = 156,44, WordPerSec = 144,39
Update = 13300, Epoch = 9, LR = 0,000520, AvgCost = 216,0970, Sent = 761, SentPerMin = 157,32, WordPerSec = 144,86
Update = 13380, Epoch = 9, LR = 0,000519, AvgCost = 214,0793, Sent = 1000, SentPerMin = 159,55, WordPerSec = 146,43
Update = 13400, Epoch = 10, LR = 0,000518, AvgCost = 208,8854, Sent = 64, SentPerMin = 132,97, WordPerSec = 115,93
Update = 13500, Epoch = 10, LR = 0,000516, AvgCost = 217,7930, Sent = 352, SentPerMin = 128,13, WordPerSec = 117,83
Update = 13600, Epoch = 10, LR = 0,000514, AvgCost = 215,1419, Sent = 645, SentPerMin = 128,75, WordPerSec = 118,51
Update = 13700, Epoch = 10, LR = 0,000513, AvgCost = 214,1985, Sent = 948, SentPerMin = 129,53, WordPerSec = 119,08
Update = 13718, Epoch = 10, LR = 0,000512, AvgCost = 213,7829, Sent = 1000, SentPerMin = 130,03, WordPerSec = 119,34
Update = 13800, Epoch = 11, LR = 0,000511, AvgCost = 214,5084, Sent = 239, SentPerMin = 161,84, WordPerSec = 147,80
Update = 13900, Epoch = 11, LR = 0,000509, AvgCost = 214,8986, Sent = 536, SentPerMin = 158,50, WordPerSec = 145,69
Update = 14000, Epoch = 11, LR = 0,000507, AvgCost = 213,3917, Sent = 831, SentPerMin = 159,59, WordPerSec = 146,54
Update = 14056, Epoch = 11, LR = 0,000506, AvgCost = 212,6693, Sent = 1000, SentPerMin = 160,40, WordPerSec = 147,21
Update = 14100, Epoch = 12, LR = 0,000505, AvgCost = 210,6487, Sent = 131, SentPerMin = 131,90, WordPerSec = 117,38
Update = 14200, Epoch = 12, LR = 0,000504, AvgCost = 214,5459, Sent = 423, SentPerMin = 128,15, WordPerSec = 117,76
Update = 14300, Epoch = 12, LR = 0,000502, AvgCost = 211,9488, Sent = 720, SentPerMin = 130,18, WordPerSec = 119,13
Update = 14394, Epoch = 12, LR = 0,000500, AvgCost = 212,0857, Sent = 1000, SentPerMin = 130,09, WordPerSec = 119,39
Starting inference...
Inference results:


)        )        )




             */

            /*
             Update = 10100, Epoch = 0, LR = 0,000597, AvgCost = 266,6917, Sent = 285, SentPerMin = 123,44, WordPerSec = 115,91
Update = 10200, Epoch = 0, LR = 0,000594, AvgCost = 247,5337, Sent = 592, SentPerMin = 124,24, WordPerSec = 113,97
Update = 10300, Epoch = 0, LR = 0,000591, AvgCost = 244,3488, Sent = 882, SentPerMin = 124,54, WordPerSec = 115,06
Update = 10338, Epoch = 0, LR = 0,000590, AvgCost = 241,2539, Sent = 1000, SentPerMin = 125,60, WordPerSec = 115,27
Update = 10400, Epoch = 1, LR = 0,000588, AvgCost = 232,5427, Sent = 178, SentPerMin = 151,52, WordPerSec = 139,81
Update = 10500, Epoch = 1, LR = 0,000586, AvgCost = 233,2050, Sent = 475, SentPerMin = 151,45, WordPerSec = 139,51
Update = 10600, Epoch = 1, LR = 0,000583, AvgCost = 231,7225, Sent = 771, SentPerMin = 151,59, WordPerSec = 140,12
Update = 10676, Epoch = 1, LR = 0,000581, AvgCost = 228,6026, Sent = 1000, SentPerMin = 153,99, WordPerSec = 141,32
Update = 10700, Epoch = 2, LR = 0,000580, AvgCost = 220,0616, Sent = 76, SentPerMin = 124,62, WordPerSec = 109,92
Update = 10800, Epoch = 2, LR = 0,000577, AvgCost = 230,4417, Sent = 363, SentPerMin = 123,47, WordPerSec = 113,86
Update = 10900, Epoch = 2, LR = 0,000575, AvgCost = 227,1132, Sent = 656, SentPerMin = 124,48, WordPerSec = 114,61
Update = 11000, Epoch = 2, LR = 0,000572, AvgCost = 224,6771, Sent = 961, SentPerMin = 125,34, WordPerSec = 114,88
Update = 11014, Epoch = 2, LR = 0,000572, AvgCost = 225,0435, Sent = 1000, SentPerMin = 125,54, WordPerSec = 115,21
Update = 11100, Epoch = 3, LR = 0,000569, AvgCost = 227,8600, Sent = 248, SentPerMin = 153,44, WordPerSec = 141,74
Update = 11200, Epoch = 3, LR = 0,000567, AvgCost = 223,8794, Sent = 549, SentPerMin = 152,59, WordPerSec = 139,81
Update = 11300, Epoch = 3, LR = 0,000564, AvgCost = 223,9433, Sent = 841, SentPerMin = 152,84, WordPerSec = 140,74
Update = 11352, Epoch = 3, LR = 0,000563, AvgCost = 222,5957, Sent = 1000, SentPerMin = 153,96, WordPerSec = 141,30
Update = 11400, Epoch = 4, LR = 0,000562, AvgCost = 220,0490, Sent = 140, SentPerMin = 127,00, WordPerSec = 114,00
Update = 11500, Epoch = 4, LR = 0,000560, AvgCost = 227,1937, Sent = 431, SentPerMin = 123,13, WordPerSec = 114,13
Update = 11600, Epoch = 4, LR = 0,000557, AvgCost = 220,2654, Sent = 732, SentPerMin = 125,47, WordPerSec = 114,72
Update = 11690, Epoch = 4, LR = 0,000555, AvgCost = 220,5285, Sent = 1000, SentPerMin = 125,96, WordPerSec = 115,61
Update = 11700, Epoch = 5, LR = 0,000555, AvgCost = 181,4536, Sent = 33, SentPerMin = 172,06, WordPerSec = 132,95
Update = 11800, Epoch = 5, LR = 0,000552, AvgCost = 226,2549, Sent = 317, SentPerMin = 153,54, WordPerSec = 142,98
Update = 11900, Epoch = 5, LR = 0,000550, AvgCost = 220,0132, Sent = 618, SentPerMin = 154,79, WordPerSec = 142,06
Update = 12000, Epoch = 5, LR = 0,000548, AvgCost = 219,5727, Sent = 914, SentPerMin = 155,40, WordPerSec = 143,20
Update = 12028, Epoch = 5, LR = 0,000547, AvgCost = 218,4449, Sent = 1000, SentPerMin = 156,44, WordPerSec = 143,58
Update = 12100, Epoch = 6, LR = 0,000545, AvgCost = 225,5246, Sent = 205, SentPerMin = 125,04, WordPerSec = 116,75
Update = 12200, Epoch = 6, LR = 0,000543, AvgCost = 218,8283, Sent = 506, SentPerMin = 125,46, WordPerSec = 115,07
Update = 12300, Epoch = 6, LR = 0,000541, AvgCost = 218,7488, Sent = 802, SentPerMin = 126,10, WordPerSec = 116,09
Update = 12366, Epoch = 6, LR = 0,000540, AvgCost = 216,8769, Sent = 1000, SentPerMin = 127,19, WordPerSec = 116,73
Update = 12400, Epoch = 7, LR = 0,000539, AvgCost = 206,6125, Sent = 106, SentPerMin = 158,69, WordPerSec = 136,68
Update = 12500, Epoch = 7, LR = 0,000537, AvgCost = 219,0541, Sent = 393, SentPerMin = 152,66, WordPerSec = 140,70
Update = 12600, Epoch = 7, LR = 0,000535, AvgCost = 215,9725, Sent = 689, SentPerMin = 156,30, WordPerSec = 143,28
Update = 12700, Epoch = 7, LR = 0,000532, AvgCost = 215,6005, Sent = 989, SentPerMin = 156,22, WordPerSec = 143,42
Update = 12704, Epoch = 7, LR = 0,000532, AvgCost = 215,4508, Sent = 1000, SentPerMin = 156,35, WordPerSec = 143,49
Update = 12800, Epoch = 8, LR = 0,000530, AvgCost = 225,7862, Sent = 275, SentPerMin = 124,79, WordPerSec = 117,14
Update = 12900, Epoch = 8, LR = 0,000528, AvgCost = 215,9585, Sent = 581, SentPerMin = 125,49, WordPerSec = 115,12
Update = 13000, Epoch = 8, LR = 0,000526, AvgCost = 216,2120, Sent = 870, SentPerMin = 126,34, WordPerSec = 116,65
Update = 13042, Epoch = 8, LR = 0,000525, AvgCost = 214,8426, Sent = 1000, SentPerMin = 127,20, WordPerSec = 116,74
Update = 13100, Epoch = 9, LR = 0,000524, AvgCost = 218,2894, Sent = 166, SentPerMin = 153,70, WordPerSec = 141,64
Update = 13200, Epoch = 9, LR = 0,000522, AvgCost = 217,9420, Sent = 463, SentPerMin = 152,99, WordPerSec = 141,21
Update = 13300, Epoch = 9, LR = 0,000520, AvgCost = 215,7507, Sent = 761, SentPerMin = 153,87, WordPerSec = 141,70
Update = 13380, Epoch = 9, LR = 0,000519, AvgCost = 213,8014, Sent = 1000, SentPerMin = 156,18, WordPerSec = 143,34
Update = 13400, Epoch = 10, LR = 0,000518, AvgCost = 210,0568, Sent = 64, SentPerMin = 129,75, WordPerSec = 113,13
Update = 13500, Epoch = 10, LR = 0,000516, AvgCost = 217,9687, Sent = 352, SentPerMin = 125,21, WordPerSec = 115,14
Update = 13600, Epoch = 10, LR = 0,000514, AvgCost = 215,0333, Sent = 645, SentPerMin = 126,00, WordPerSec = 115,98
Update = 13700, Epoch = 10, LR = 0,000513, AvgCost = 213,9785, Sent = 948, SentPerMin = 126,70, WordPerSec = 116,47
Update = 13718, Epoch = 10, LR = 0,000512, AvgCost = 213,5403, Sent = 1000, SentPerMin = 127,22, WordPerSec = 116,75
Update = 13800, Epoch = 11, LR = 0,000511, AvgCost = 214,9280, Sent = 239, SentPerMin = 157,96, WordPerSec = 144,25
Update = 13900, Epoch = 11, LR = 0,000509, AvgCost = 215,0342, Sent = 536, SentPerMin = 154,33, WordPerSec = 141,86
Update = 14000, Epoch = 11, LR = 0,000507, AvgCost = 213,3008, Sent = 831, SentPerMin = 155,28, WordPerSec = 142,59
Update = 14056, Epoch = 11, LR = 0,000506, AvgCost = 212,5923, Sent = 1000, SentPerMin = 156,10, WordPerSec = 143,26
Update = 14100, Epoch = 12, LR = 0,000505, AvgCost = 210,9409, Sent = 131, SentPerMin = 129,11, WordPerSec = 114,90
Update = 14200, Epoch = 12, LR = 0,000504, AvgCost = 215,0082, Sent = 423, SentPerMin = 125,24, WordPerSec = 115,09
Update = 14300, Epoch = 12, LR = 0,000502, AvgCost = 211,9933, Sent = 720, SentPerMin = 127,07, WordPerSec = 116,29
Update = 14394, Epoch = 12, LR = 0,000500, AvgCost = 212,0777, Sent = 1000, SentPerMin = 127,06, WordPerSec = 116,61
Update = 14400, Epoch = 13, LR = 0,000500, AvgCost = 209,4849, Sent = 16, SentPerMin = 152,63, WordPerSec = 148,97
Update = 14500, Epoch = 13, LR = 0,000498, AvgCost = 219,7876, Sent = 304, SentPerMin = 154,39, WordPerSec = 144,20
Update = 14600, Epoch = 13, LR = 0,000497, AvgCost = 213,0103, Sent = 606, SentPerMin = 154,29, WordPerSec = 141,80
Update = 14700, Epoch = 13, LR = 0,000495, AvgCost = 212,4569, Sent = 902, SentPerMin = 155,11, WordPerSec = 142,85
Update = 14732, Epoch = 13, LR = 0,000494, AvgCost = 211,1812, Sent = 1000, SentPerMin = 156,15, WordPerSec = 143,31
Update = 14800, Epoch = 14, LR = 0,000493, AvgCost = 216,5257, Sent = 195, SentPerMin = 126,71, WordPerSec = 116,47
Update = 14900, Epoch = 14, LR = 0,000492, AvgCost = 212,6616, Sent = 495, SentPerMin = 126,36, WordPerSec = 115,48
Update = 15000, Epoch = 14, LR = 0,000490, AvgCost = 213,2868, Sent = 789, SentPerMin = 125,67, WordPerSec = 115,87
Update = 15070, Epoch = 14, LR = 0,000489, AvgCost = 211,0486, Sent = 1000, SentPerMin = 127,10, WordPerSec = 116,64
Update = 15100, Epoch = 15, LR = 0,000488, AvgCost = 207,5085, Sent = 92, SentPerMin = 153,94, WordPerSec = 136,62
Update = 15200, Epoch = 15, LR = 0,000487, AvgCost = 214,0518, Sent = 381, SentPerMin = 152,97, WordPerSec = 140,76
Update = 15300, Epoch = 15, LR = 0,000485, AvgCost = 211,2433, Sent = 676, SentPerMin = 155,20, WordPerSec = 142,49
Update = 15400, Epoch = 15, LR = 0,000483, AvgCost = 209,0473, Sent = 979, SentPerMin = 156,25, WordPerSec = 143,07
Update = 15408, Epoch = 15, LR = 0,000483, AvgCost = 210,0960, Sent = 1000, SentPerMin = 156,06, WordPerSec = 143,23
Update = 15500, Epoch = 16, LR = 0,000482, AvgCost = 217,3832, Sent = 265, SentPerMin = 126,14, WordPerSec = 117,06
Update = 15600, Epoch = 16, LR = 0,000480, AvgCost = 211,7477, Sent = 566, SentPerMin = 125,61, WordPerSec = 115,60
Update = 15700, Epoch = 16, LR = 0,000479, AvgCost = 210,8502, Sent = 859, SentPerMin = 126,07, WordPerSec = 116,18
Update = 15746, Epoch = 16, LR = 0,000478, AvgCost = 209,4910, Sent = 1000, SentPerMin = 127,04, WordPerSec = 116,59
Update = 15800, Epoch = 17, LR = 0,000477, AvgCost = 211,2095, Sent = 157, SentPerMin = 154,90, WordPerSec = 140,91
Update = 15900, Epoch = 17, LR = 0,000476, AvgCost = 213,3586, Sent = 450, SentPerMin = 153,81, WordPerSec = 141,83
Update = 16000, Epoch = 17, LR = 0,000474, AvgCost = 208,6196, Sent = 752, SentPerMin = 154,41, WordPerSec = 141,57
Update = 16084, Epoch = 17, LR = 0,000473, AvgCost = 208,3930, Sent = 1000, SentPerMin = 156,02, WordPerSec = 143,19
Update = 16100, Epoch = 18, LR = 0,000473, AvgCost = 200,2530, Sent = 51, SentPerMin = 132,22, WordPerSec = 113,98
Update = 16200, Epoch = 18, LR = 0,000471, AvgCost = 214,2693, Sent = 336, SentPerMin = 125,48, WordPerSec = 116,26
Update = 16300, Epoch = 18, LR = 0,000470, AvgCost = 208,8546, Sent = 636, SentPerMin = 125,99, WordPerSec = 115,43
Update = 16400, Epoch = 18, LR = 0,000469, AvgCost = 208,5599, Sent = 935, SentPerMin = 126,31, WordPerSec = 116,11
Update = 16422, Epoch = 18, LR = 0,000468, AvgCost = 208,0487, Sent = 1000, SentPerMin = 127,03, WordPerSec = 116,58
Update = 16500, Epoch = 19, LR = 0,000467, AvgCost = 210,6055, Sent = 225, SentPerMin = 157,04, WordPerSec = 144,36
Update = 16600, Epoch = 19, LR = 0,000466, AvgCost = 210,2456, Sent = 524, SentPerMin = 153,96, WordPerSec = 141,67
Update = 16700, Epoch = 19, LR = 0,000464, AvgCost = 208,4390, Sent = 819, SentPerMin = 155,08, WordPerSec = 142,64
Update = 16760, Epoch = 19, LR = 0,000463, AvgCost = 207,0240, Sent = 1000, SentPerMin = 156,22, WordPerSec = 143,38
Update = 16800, Epoch = 20, LR = 0,000463, AvgCost = 202,3210, Sent = 122, SentPerMin = 130,09, WordPerSec = 113,55
Update = 16900, Epoch = 20, LR = 0,000462, AvgCost = 209,9734, Sent = 411, SentPerMin = 124,70, WordPerSec = 114,85
Update = 17000, Epoch = 20, LR = 0,000460, AvgCost = 206,3598, Sent = 709, SentPerMin = 127,34, WordPerSec = 116,32
Update = 17098, Epoch = 20, LR = 0,000459, AvgCost = 206,7750, Sent = 1000, SentPerMin = 127,03, WordPerSec = 116,58
Update = 17100, Epoch = 21, LR = 0,000459, AvgCost = 181,2885, Sent = 6, SentPerMin = 168,78, WordPerSec = 153,30
Update = 17200, Epoch = 21, LR = 0,000457, AvgCost = 214,3475, Sent = 292, SentPerMin = 153,88, WordPerSec = 143,97
Update = 17300, Epoch = 21, LR = 0,000456, AvgCost = 207,9620, Sent = 595, SentPerMin = 153,94, WordPerSec = 141,69
Update = 17400, Epoch = 21, LR = 0,000455, AvgCost = 207,2251, Sent = 890, SentPerMin = 154,99, WordPerSec = 142,87
Update = 17436, Epoch = 21, LR = 0,000454, AvgCost = 205,7156, Sent = 1000, SentPerMin = 156,04, WordPerSec = 143,21
Update = 17500, Epoch = 22, LR = 0,000454, AvgCost = 211,1530, Sent = 184, SentPerMin = 126,42, WordPerSec = 116,15
Update = 17600, Epoch = 22, LR = 0,000452, AvgCost = 208,3593, Sent = 482, SentPerMin = 125,62, WordPerSec = 115,49
Update = 17700, Epoch = 22, LR = 0,000451, AvgCost = 208,8443, Sent = 775, SentPerMin = 124,98, WordPerSec = 115,80
Update = 17774, Epoch = 22, LR = 0,000450, AvgCost = 205,7936, Sent = 1000, SentPerMin = 126,98, WordPerSec = 116,54
Update = 17800, Epoch = 23, LR = 0,000450, AvgCost = 198,6427, Sent = 81, SentPerMin = 153,86, WordPerSec = 136,57
Update = 17900, Epoch = 23, LR = 0,000448, AvgCost = 209,4839, Sent = 368, SentPerMin = 151,99, WordPerSec = 140,66
Update = 18000, Epoch = 23, LR = 0,000447, AvgCost = 205,8855, Sent = 663, SentPerMin = 154,84, WordPerSec = 142,26
Update = 18100, Epoch = 23, LR = 0,000446, AvgCost = 203,8687, Sent = 969, SentPerMin = 156,02, WordPerSec = 142,67
Update = 18112, Epoch = 23, LR = 0,000446, AvgCost = 204,6649, Sent = 1000, SentPerMin = 155,92, WordPerSec = 143,10
Update = 18200, Epoch = 24, LR = 0,000445, AvgCost = 214,3434, Sent = 251, SentPerMin = 125,36, WordPerSec = 117,12
Update = 18300, Epoch = 24, LR = 0,000444, AvgCost = 205,6699, Sent = 556, SentPerMin = 126,29, WordPerSec = 115,60
Update = 18400, Epoch = 24, LR = 0,000442, AvgCost = 206,2061, Sent = 847, SentPerMin = 126,06, WordPerSec = 116,19
Update = 18450, Epoch = 24, LR = 0,000442, AvgCost = 204,7313, Sent = 1000, SentPerMin = 126,90, WordPerSec = 116,46
Update = 18500, Epoch = 25, LR = 0,000441, AvgCost = 206,7782, Sent = 145, SentPerMin = 153,30, WordPerSec = 139,43
Update = 18600, Epoch = 25, LR = 0,000440, AvgCost = 209,3469, Sent = 437, SentPerMin = 152,36, WordPerSec = 141,19
Update = 18700, Epoch = 25, LR = 0,000439, AvgCost = 203,6330, Sent = 739, SentPerMin = 153,96, WordPerSec = 141,14
Update = 18788, Epoch = 25, LR = 0,000438, AvgCost = 203,5250, Sent = 1000, SentPerMin = 155,36, WordPerSec = 142,59
Starting inference...
Inference results:
,在和在
, 的的 的 的。
和的的的的的的。
,在在的
,在

             */

            /*
             Update = 10100, Epoch = 0, LR = 0,000597, AvgCost = 266,6059, Sent = 285, SentPerMin = 128,38, WordPerSec = 120,55
Update = 10200, Epoch = 0, LR = 0,000594, AvgCost = 248,0226, Sent = 592, SentPerMin = 130,34, WordPerSec = 119,56
Update = 10300, Epoch = 0, LR = 0,000591, AvgCost = 245,1151, Sent = 882, SentPerMin = 130,92, WordPerSec = 120,95
Update = 10338, Epoch = 0, LR = 0,000590, AvgCost = 241,9874, Sent = 1000, SentPerMin = 132,08, WordPerSec = 121,22
Update = 10400, Epoch = 1, LR = 0,000588, AvgCost = 232,6085, Sent = 178, SentPerMin = 160,17, WordPerSec = 147,80
Update = 10500, Epoch = 1, LR = 0,000586, AvgCost = 233,4318, Sent = 475, SentPerMin = 159,97, WordPerSec = 147,35
Update = 10600, Epoch = 1, LR = 0,000583, AvgCost = 231,9244, Sent = 771, SentPerMin = 159,95, WordPerSec = 147,84
Update = 10676, Epoch = 1, LR = 0,000581, AvgCost = 228,6890, Sent = 1000, SentPerMin = 162,54, WordPerSec = 149,17
Update = 10700, Epoch = 2, LR = 0,000580, AvgCost = 219,9773, Sent = 76, SentPerMin = 131,72, WordPerSec = 116,18
Update = 10800, Epoch = 2, LR = 0,000577, AvgCost = 230,4425, Sent = 363, SentPerMin = 130,65, WordPerSec = 120,48
Update = 10900, Epoch = 2, LR = 0,000575, AvgCost = 227,1563, Sent = 656, SentPerMin = 131,61, WordPerSec = 121,17
Update = 11000, Epoch = 2, LR = 0,000572, AvgCost = 224,4940, Sent = 961, SentPerMin = 132,52, WordPerSec = 121,46
Update = 11014, Epoch = 2, LR = 0,000572, AvgCost = 224,8960, Sent = 1000, SentPerMin = 132,75, WordPerSec = 121,83
Update = 11100, Epoch = 3, LR = 0,000569, AvgCost = 228,2183, Sent = 248, SentPerMin = 162,56, WordPerSec = 150,17
Update = 11200, Epoch = 3, LR = 0,000567, AvgCost = 224,0640, Sent = 549, SentPerMin = 161,72, WordPerSec = 148,17
Update = 11300, Epoch = 3, LR = 0,000564, AvgCost = 223,9447, Sent = 841, SentPerMin = 162,14, WordPerSec = 149,31
Update = 11352, Epoch = 3, LR = 0,000563, AvgCost = 222,5438, Sent = 1000, SentPerMin = 163,28, WordPerSec = 149,85
Update = 11400, Epoch = 4, LR = 0,000562, AvgCost = 220,8745, Sent = 140, SentPerMin = 134,27, WordPerSec = 120,53
Update = 11500, Epoch = 4, LR = 0,000560, AvgCost = 227,6576, Sent = 431, SentPerMin = 130,25, WordPerSec = 120,74
Update = 11600, Epoch = 4, LR = 0,000557, AvgCost = 220,7120, Sent = 732, SentPerMin = 132,54, WordPerSec = 121,18
Update = 11690, Epoch = 4, LR = 0,000555, AvgCost = 220,9754, Sent = 1000, SentPerMin = 132,59, WordPerSec = 121,69
Update = 11700, Epoch = 5, LR = 0,000555, AvgCost = 182,8448, Sent = 33, SentPerMin = 179,75, WordPerSec = 138,90
Update = 11800, Epoch = 5, LR = 0,000552, AvgCost = 227,2231, Sent = 317, SentPerMin = 160,14, WordPerSec = 149,12
Update = 11900, Epoch = 5, LR = 0,000550, AvgCost = 221,0142, Sent = 618, SentPerMin = 161,38, WordPerSec = 148,11
Update = 12000, Epoch = 5, LR = 0,000548, AvgCost = 220,6514, Sent = 914, SentPerMin = 161,93, WordPerSec = 149,21
Update = 12028, Epoch = 5, LR = 0,000547, AvgCost = 219,4818, Sent = 1000, SentPerMin = 162,96, WordPerSec = 149,56
Update = 12100, Epoch = 6, LR = 0,000545, AvgCost = 227,0088, Sent = 205, SentPerMin = 130,57, WordPerSec = 121,91
Update = 12200, Epoch = 6, LR = 0,000543, AvgCost = 220,6343, Sent = 506, SentPerMin = 131,05, WordPerSec = 120,20
Update = 12300, Epoch = 6, LR = 0,000541, AvgCost = 220,5943, Sent = 802, SentPerMin = 131,64, WordPerSec = 121,20
Update = 12366, Epoch = 6, LR = 0,000540, AvgCost = 218,7422, Sent = 1000, SentPerMin = 132,75, WordPerSec = 121,84
Update = 12400, Epoch = 7, LR = 0,000539, AvgCost = 208,9023, Sent = 106, SentPerMin = 165,84, WordPerSec = 142,84
Update = 12500, Epoch = 7, LR = 0,000537, AvgCost = 221,6145, Sent = 393, SentPerMin = 159,83, WordPerSec = 147,30
Update = 12600, Epoch = 7, LR = 0,000535, AvgCost = 218,2485, Sent = 689, SentPerMin = 163,40, WordPerSec = 149,78
Update = 12700, Epoch = 7, LR = 0,000532, AvgCost = 217,7586, Sent = 989, SentPerMin = 163,34, WordPerSec = 149,94
Update = 12704, Epoch = 7, LR = 0,000532, AvgCost = 217,6150, Sent = 1000, SentPerMin = 163,45, WordPerSec = 150,01
Update = 12800, Epoch = 8, LR = 0,000530, AvgCost = 228,4162, Sent = 275, SentPerMin = 130,15, WordPerSec = 122,17
Update = 12900, Epoch = 8, LR = 0,000528, AvgCost = 218,3649, Sent = 581, SentPerMin = 130,93, WordPerSec = 120,11
Update = 13000, Epoch = 8, LR = 0,000526, AvgCost = 218,4519, Sent = 870, SentPerMin = 131,71, WordPerSec = 121,61
Update = 13042, Epoch = 8, LR = 0,000525, AvgCost = 216,9163, Sent = 1000, SentPerMin = 132,60, WordPerSec = 121,70
Update = 13100, Epoch = 9, LR = 0,000524, AvgCost = 220,1015, Sent = 166, SentPerMin = 160,79, WordPerSec = 148,17
Update = 13200, Epoch = 9, LR = 0,000522, AvgCost = 219,9542, Sent = 463, SentPerMin = 159,93, WordPerSec = 147,62
Update = 13300, Epoch = 9, LR = 0,000520, AvgCost = 217,6149, Sent = 761, SentPerMin = 160,73, WordPerSec = 148,01
Update = 13380, Epoch = 9, LR = 0,000519, AvgCost = 215,4726, Sent = 1000, SentPerMin = 163,05, WordPerSec = 149,64
Update = 13400, Epoch = 10, LR = 0,000518, AvgCost = 211,8173, Sent = 64, SentPerMin = 135,44, WordPerSec = 118,09
Update = 13500, Epoch = 10, LR = 0,000516, AvgCost = 219,5116, Sent = 352, SentPerMin = 130,67, WordPerSec = 120,16
Update = 13600, Epoch = 10, LR = 0,000514, AvgCost = 216,3788, Sent = 645, SentPerMin = 131,38, WordPerSec = 120,93
Update = 13700, Epoch = 10, LR = 0,000513, AvgCost = 215,0072, Sent = 948, SentPerMin = 132,07, WordPerSec = 121,41
Update = 13718, Epoch = 10, LR = 0,000512, AvgCost = 214,5626, Sent = 1000, SentPerMin = 132,60, WordPerSec = 121,69
Update = 13800, Epoch = 11, LR = 0,000511, AvgCost = 216,3784, Sent = 239, SentPerMin = 164,19, WordPerSec = 149,95
Update = 13900, Epoch = 11, LR = 0,000509, AvgCost = 215,9791, Sent = 536, SentPerMin = 160,68, WordPerSec = 147,70
Update = 14000, Epoch = 11, LR = 0,000507, AvgCost = 214,2210, Sent = 831, SentPerMin = 161,70, WordPerSec = 148,48
Update = 14056, Epoch = 11, LR = 0,000506, AvgCost = 213,3698, Sent = 1000, SentPerMin = 162,55, WordPerSec = 149,18
Update = 14100, Epoch = 12, LR = 0,000505, AvgCost = 211,6606, Sent = 131, SentPerMin = 134,25, WordPerSec = 119,47
Update = 14200, Epoch = 12, LR = 0,000504, AvgCost = 215,7425, Sent = 423, SentPerMin = 130,64, WordPerSec = 120,06
Update = 14300, Epoch = 12, LR = 0,000502, AvgCost = 212,6607, Sent = 720, SentPerMin = 132,51, WordPerSec = 121,27
Update = 14394, Epoch = 12, LR = 0,000500, AvgCost = 212,7915, Sent = 1000, SentPerMin = 132,51, WordPerSec = 121,61
Update = 14400, Epoch = 13, LR = 0,000500, AvgCost = 209,7390, Sent = 16, SentPerMin = 160,49, WordPerSec = 156,64
Update = 14500, Epoch = 13, LR = 0,000498, AvgCost = 220,3474, Sent = 304, SentPerMin = 160,95, WordPerSec = 150,33
Update = 14600, Epoch = 13, LR = 0,000497, AvgCost = 213,6989, Sent = 606, SentPerMin = 160,95, WordPerSec = 147,93
Update = 14700, Epoch = 13, LR = 0,000495, AvgCost = 213,2233, Sent = 902, SentPerMin = 161,83, WordPerSec = 149,04
Update = 14732, Epoch = 13, LR = 0,000494, AvgCost = 211,8864, Sent = 1000, SentPerMin = 162,90, WordPerSec = 149,50
Update = 14800, Epoch = 14, LR = 0,000493, AvgCost = 216,8440, Sent = 195, SentPerMin = 132,26, WordPerSec = 121,57
Update = 14900, Epoch = 14, LR = 0,000492, AvgCost = 213,0098, Sent = 495, SentPerMin = 131,86, WordPerSec = 120,51
Update = 15000, Epoch = 14, LR = 0,000490, AvgCost = 213,9693, Sent = 789, SentPerMin = 131,11, WordPerSec = 120,89
Update = 15070, Epoch = 14, LR = 0,000489, AvgCost = 211,5583, Sent = 1000, SentPerMin = 132,59, WordPerSec = 121,69
Update = 15100, Epoch = 15, LR = 0,000488, AvgCost = 207,2994, Sent = 92, SentPerMin = 160,09, WordPerSec = 142,08
Update = 15200, Epoch = 15, LR = 0,000487, AvgCost = 214,2991, Sent = 381, SentPerMin = 159,88, WordPerSec = 147,12
Update = 15300, Epoch = 15, LR = 0,000485, AvgCost = 211,8647, Sent = 676, SentPerMin = 162,36, WordPerSec = 149,06
Update = 15400, Epoch = 15, LR = 0,000483, AvgCost = 209,5406, Sent = 979, SentPerMin = 163,33, WordPerSec = 149,55
Update = 15408, Epoch = 15, LR = 0,000483, AvgCost = 210,5714, Sent = 1000, SentPerMin = 163,12, WordPerSec = 149,70
Update = 15500, Epoch = 16, LR = 0,000482, AvgCost = 217,3095, Sent = 265, SentPerMin = 131,81, WordPerSec = 122,32
Update = 15600, Epoch = 16, LR = 0,000480, AvgCost = 211,9204, Sent = 566, SentPerMin = 130,98, WordPerSec = 120,55
Update = 15700, Epoch = 16, LR = 0,000479, AvgCost = 211,1936, Sent = 859, SentPerMin = 131,38, WordPerSec = 121,06
Update = 15746, Epoch = 16, LR = 0,000478, AvgCost = 209,7767, Sent = 1000, SentPerMin = 132,41, WordPerSec = 121,52
Update = 15800, Epoch = 17, LR = 0,000477, AvgCost = 211,1617, Sent = 157, SentPerMin = 161,01, WordPerSec = 146,47
Update = 15900, Epoch = 17, LR = 0,000476, AvgCost = 213,5695, Sent = 450, SentPerMin = 160,03, WordPerSec = 147,56
Update = 16000, Epoch = 17, LR = 0,000474, AvgCost = 209,1908, Sent = 752, SentPerMin = 161,07, WordPerSec = 147,67
Update = 16084, Epoch = 17, LR = 0,000473, AvgCost = 209,0055, Sent = 1000, SentPerMin = 162,83, WordPerSec = 149,44
Update = 16100, Epoch = 18, LR = 0,000473, AvgCost = 202,0980, Sent = 51, SentPerMin = 137,91, WordPerSec = 118,89
Update = 16200, Epoch = 18, LR = 0,000471, AvgCost = 215,4586, Sent = 336, SentPerMin = 130,71, WordPerSec = 121,10
Update = 16300, Epoch = 18, LR = 0,000470, AvgCost = 209,4715, Sent = 636, SentPerMin = 131,26, WordPerSec = 120,27
Update = 16400, Epoch = 18, LR = 0,000469, AvgCost = 209,1325, Sent = 935, SentPerMin = 131,05, WordPerSec = 120,47
Update = 16422, Epoch = 18, LR = 0,000468, AvgCost = 208,6749, Sent = 1000, SentPerMin = 131,75, WordPerSec = 120,91
Update = 16500, Epoch = 19, LR = 0,000467, AvgCost = 211,6414, Sent = 225, SentPerMin = 157,72, WordPerSec = 144,99
Update = 16600, Epoch = 19, LR = 0,000466, AvgCost = 210,9279, Sent = 524, SentPerMin = 157,86, WordPerSec = 145,25
Update = 16700, Epoch = 19, LR = 0,000464, AvgCost = 209,0055, Sent = 819, SentPerMin = 159,86, WordPerSec = 147,04
Update = 16760, Epoch = 19, LR = 0,000463, AvgCost = 207,6659, Sent = 1000, SentPerMin = 161,38, WordPerSec = 148,11
Update = 16800, Epoch = 20, LR = 0,000463, AvgCost = 203,9848, Sent = 122, SentPerMin = 135,92, WordPerSec = 118,63
Update = 16900, Epoch = 20, LR = 0,000462, AvgCost = 210,9296, Sent = 411, SentPerMin = 130,03, WordPerSec = 119,76
Update = 17000, Epoch = 20, LR = 0,000460, AvgCost = 206,9427, Sent = 709, SentPerMin = 132,75, WordPerSec = 121,26
Update = 17098, Epoch = 20, LR = 0,000459, AvgCost = 207,4486, Sent = 1000, SentPerMin = 132,35, WordPerSec = 121,47
Update = 17100, Epoch = 21, LR = 0,000459, AvgCost = 183,1378, Sent = 6, SentPerMin = 177,72, WordPerSec = 161,43
Update = 17200, Epoch = 21, LR = 0,000457, AvgCost = 215,8214, Sent = 292, SentPerMin = 160,01, WordPerSec = 149,71
Update = 17300, Epoch = 21, LR = 0,000456, AvgCost = 208,8154, Sent = 595, SentPerMin = 160,02, WordPerSec = 147,28
Update = 17400, Epoch = 21, LR = 0,000455, AvgCost = 207,9716, Sent = 890, SentPerMin = 161,14, WordPerSec = 148,54
Update = 17436, Epoch = 21, LR = 0,000454, AvgCost = 206,6101, Sent = 1000, SentPerMin = 162,28, WordPerSec = 148,93
Update = 17500, Epoch = 22, LR = 0,000454, AvgCost = 213,1795, Sent = 184, SentPerMin = 132,19, WordPerSec = 121,45
Update = 17600, Epoch = 22, LR = 0,000452, AvgCost = 209,4238, Sent = 482, SentPerMin = 131,03, WordPerSec = 120,46
Update = 17700, Epoch = 22, LR = 0,000451, AvgCost = 209,6422, Sent = 775, SentPerMin = 130,33, WordPerSec = 120,76
Update = 17774, Epoch = 22, LR = 0,000450, AvgCost = 206,3572, Sent = 1000, SentPerMin = 132,34, WordPerSec = 121,46
Update = 17800, Epoch = 23, LR = 0,000450, AvgCost = 200,9257, Sent = 81, SentPerMin = 159,78, WordPerSec = 141,83
Update = 17900, Epoch = 23, LR = 0,000448, AvgCost = 210,8819, Sent = 368, SentPerMin = 158,02, WordPerSec = 146,24
Update = 18000, Epoch = 23, LR = 0,000447, AvgCost = 206,6538, Sent = 663, SentPerMin = 160,95, WordPerSec = 147,87
Update = 18100, Epoch = 23, LR = 0,000446, AvgCost = 204,4698, Sent = 969, SentPerMin = 162,12, WordPerSec = 148,24
Update = 18112, Epoch = 23, LR = 0,000446, AvgCost = 205,3462, Sent = 1000, SentPerMin = 162,01, WordPerSec = 148,68
Update = 18200, Epoch = 24, LR = 0,000445, AvgCost = 214,7968, Sent = 251, SentPerMin = 131,10, WordPerSec = 122,47
Update = 18300, Epoch = 24, LR = 0,000444, AvgCost = 206,0736, Sent = 556, SentPerMin = 131,66, WordPerSec = 120,51
Update = 18400, Epoch = 24, LR = 0,000442, AvgCost = 206,2661, Sent = 847, SentPerMin = 131,37, WordPerSec = 121,09
Update = 18450, Epoch = 24, LR = 0,000442, AvgCost = 204,9973, Sent = 1000, SentPerMin = 132,23, WordPerSec = 121,36
Update = 18500, Epoch = 25, LR = 0,000441, AvgCost = 207,1047, Sent = 145, SentPerMin = 160,63, WordPerSec = 146,10
Update = 18600, Epoch = 25, LR = 0,000440, AvgCost = 209,6360, Sent = 437, SentPerMin = 159,55, WordPerSec = 147,86
Update = 18700, Epoch = 25, LR = 0,000439, AvgCost = 203,8211, Sent = 739, SentPerMin = 161,07, WordPerSec = 147,65
Update = 18788, Epoch = 25, LR = 0,000438, AvgCost = 203,8889, Sent = 1000, SentPerMin = 162,50, WordPerSec = 149,14
Update = 18800, Epoch = 26, LR = 0,000438, AvgCost = 166,3123, Sent = 41, SentPerMin = 145,19, WordPerSec = 111,79
Update = 18900, Epoch = 26, LR = 0,000436, AvgCost = 210,0808, Sent = 324, SentPerMin = 130,45, WordPerSec = 121,17
Update = 19000, Epoch = 26, LR = 0,000435, AvgCost = 204,8132, Sent = 624, SentPerMin = 131,15, WordPerSec = 120,23
Update = 19100, Epoch = 26, LR = 0,000434, AvgCost = 204,4068, Sent = 921, SentPerMin = 131,81, WordPerSec = 121,27
Update = 19126, Epoch = 26, LR = 0,000434, AvgCost = 203,6870, Sent = 1000, SentPerMin = 132,27, WordPerSec = 121,40
Update = 19200, Epoch = 27, LR = 0,000433, AvgCost = 207,9200, Sent = 213, SentPerMin = 161,97, WordPerSec = 149,43
Update = 19300, Epoch = 27, LR = 0,000432, AvgCost = 206,3627, Sent = 511, SentPerMin = 159,19, WordPerSec = 146,79
Update = 19400, Epoch = 27, LR = 0,000431, AvgCost = 204,3973, Sent = 808, SentPerMin = 161,12, WordPerSec = 148,41
Update = 19464, Epoch = 27, LR = 0,000430, AvgCost = 202,6744, Sent = 1000, SentPerMin = 162,57, WordPerSec = 149,20
Update = 19500, Epoch = 28, LR = 0,000430, AvgCost = 191,6707, Sent = 113, SentPerMin = 136,46, WordPerSec = 116,90
Update = 19600, Epoch = 28, LR = 0,000429, AvgCost = 205,4840, Sent = 399, SentPerMin = 129,86, WordPerSec = 119,53
Update = 19700, Epoch = 28, LR = 0,000427, AvgCost = 203,2269, Sent = 696, SentPerMin = 131,90, WordPerSec = 120,85
Update = 19800, Epoch = 28, LR = 0,000426, AvgCost = 202,6739, Sent = 995, SentPerMin = 132,26, WordPerSec = 121,30
Update = 19802, Epoch = 28, LR = 0,000426, AvgCost = 202,6267, Sent = 1000, SentPerMin = 132,21, WordPerSec = 121,34
Update = 19900, Epoch = 29, LR = 0,000425, AvgCost = 211,3882, Sent = 280, SentPerMin = 159,71, WordPerSec = 149,87
Update = 20000, Epoch = 29, LR = 0,000424, AvgCost = 202,4338, Sent = 588, SentPerMin = 160,96, WordPerSec = 147,24
Update = 20100, Epoch = 29, LR = 0,000423, AvgCost = 202,7752, Sent = 877, SentPerMin = 131,12, WordPerSec = 120,92
Update = 20140, Epoch = 29, LR = 0,000423, AvgCost = 201,6161, Sent = 1000, SentPerMin = 135,15, WordPerSec = 124,03
Update = 20200, Epoch = 30, LR = 0,000422, AvgCost = 204,8095, Sent = 173, SentPerMin = 159,93, WordPerSec = 147,14
Update = 20300, Epoch = 30, LR = 0,000421, AvgCost = 205,6892, Sent = 469, SentPerMin = 141,89, WordPerSec = 130,88
Update = 20400, Epoch = 30, LR = 0,000420, AvgCost = 203,9748, Sent = 766, SentPerMin = 137,41, WordPerSec = 126,80
Update = 20478, Epoch = 30, LR = 0,000419, AvgCost = 201,6889, Sent = 1000, SentPerMin = 137,66, WordPerSec = 126,34
Update = 20500, Epoch = 31, LR = 0,000419, AvgCost = 195,1593, Sent = 71, SentPerMin = 134,23, WordPerSec = 116,17
Update = 20600, Epoch = 31, LR = 0,000418, AvgCost = 204,6913, Sent = 358, SentPerMin = 141,32, WordPerSec = 129,81
Update = 20700, Epoch = 31, LR = 0,000417, AvgCost = 202,1864, Sent = 652, SentPerMin = 150,37, WordPerSec = 138,08
Update = 20800, Epoch = 31, LR = 0,000416, AvgCost = 200,7946, Sent = 955, SentPerMin = 154,40, WordPerSec = 141,65
Update = 20816, Epoch = 31, LR = 0,000416, AvgCost = 200,8296, Sent = 1000, SentPerMin = 155,15, WordPerSec = 142,39
Update = 20900, Epoch = 32, LR = 0,000415, AvgCost = 202,2176, Sent = 244, SentPerMin = 159,91, WordPerSec = 146,10
Update = 21000, Epoch = 32, LR = 0,000414, AvgCost = 202,0560, Sent = 542, SentPerMin = 141,16, WordPerSec = 129,65
Update = 21100, Epoch = 32, LR = 0,000413, AvgCost = 201,4883, Sent = 835, SentPerMin = 138,07, WordPerSec = 127,03
Update = 21154, Epoch = 32, LR = 0,000413, AvgCost = 200,4801, Sent = 1000, SentPerMin = 137,60, WordPerSec = 126,29
Update = 21200, Epoch = 33, LR = 0,000412, AvgCost = 198,2401, Sent = 136, SentPerMin = 134,77, WordPerSec = 119,92
Update = 21300, Epoch = 33, LR = 0,000411, AvgCost = 202,1168, Sent = 428, SentPerMin = 144,03, WordPerSec = 132,48
Update = 21400, Epoch = 33, LR = 0,000410, AvgCost = 199,8716, Sent = 725, SentPerMin = 151,97, WordPerSec = 139,31
Update = 21492, Epoch = 33, LR = 0,000409, AvgCost = 199,7353, Sent = 1000, SentPerMin = 155,22, WordPerSec = 142,46
Update = 21500, Epoch = 34, LR = 0,000409, AvgCost = 182,6836, Sent = 23, SentPerMin = 163,04, WordPerSec = 142,13
Update = 21600, Epoch = 34, LR = 0,000408, AvgCost = 205,3836, Sent = 311, SentPerMin = 149,27, WordPerSec = 138,78
Update = 21700, Epoch = 34, LR = 0,000407, AvgCost = 201,2832, Sent = 611, SentPerMin = 139,21, WordPerSec = 128,21
Update = 21800, Epoch = 34, LR = 0,000406, AvgCost = 200,7974, Sent = 906, SentPerMin = 137,22, WordPerSec = 126,59
Update = 21830, Epoch = 34, LR = 0,000406, AvgCost = 199,4442, Sent = 1000, SentPerMin = 137,46, WordPerSec = 126,16
Update = 21900, Epoch = 35, LR = 0,000405, AvgCost = 202,9926, Sent = 200, SentPerMin = 131,95, WordPerSec = 121,09
Update = 22000, Epoch = 35, LR = 0,000405, AvgCost = 200,1459, Sent = 501, SentPerMin = 146,89, WordPerSec = 134,57
Update = 22100, Epoch = 35, LR = 0,000404, AvgCost = 200,4139, Sent = 795, SentPerMin = 151,76, WordPerSec = 139,83
Update = 22168, Epoch = 35, LR = 0,000403, AvgCost = 198,5685, Sent = 1000, SentPerMin = 155,08, WordPerSec = 142,33
Update = 22200, Epoch = 36, LR = 0,000403, AvgCost = 192,8358, Sent = 99, SentPerMin = 163,93, WordPerSec = 142,84
Update = 22300, Epoch = 36, LR = 0,000402, AvgCost = 200,0402, Sent = 388, SentPerMin = 145,08, WordPerSec = 132,98
Update = 22400, Epoch = 36, LR = 0,000401, AvgCost = 198,7063, Sent = 683, SentPerMin = 139,77, WordPerSec = 128,07
Update = 22500, Epoch = 36, LR = 0,000400, AvgCost = 198,4758, Sent = 984, SentPerMin = 137,43, WordPerSec = 126,12
Update = 22506, Epoch = 36, LR = 0,000400, AvgCost = 198,3805, Sent = 1000, SentPerMin = 137,44, WordPerSec = 126,14
Update = 22600, Epoch = 37, LR = 0,000399, AvgCost = 206,0819, Sent = 269, SentPerMin = 136,62, WordPerSec = 127,70
Update = 22700, Epoch = 37, LR = 0,000398, AvgCost = 199,2982, Sent = 573, SentPerMin = 148,34, WordPerSec = 136,24
Update = 22800, Epoch = 37, LR = 0,000397, AvgCost = 198,6906, Sent = 865, SentPerMin = 152,06, WordPerSec = 140,05
Update = 22844, Epoch = 37, LR = 0,000397, AvgCost = 197,8552, Sent = 1000, SentPerMin = 154,13, WordPerSec = 141,45
Update = 22900, Epoch = 38, LR = 0,000396, AvgCost = 201,2718, Sent = 161, SentPerMin = 158,66, WordPerSec = 145,54
Update = 23000, Epoch = 38, LR = 0,000396, AvgCost = 202,1062, Sent = 456, SentPerMin = 140,54, WordPerSec = 129,85
Update = 23100, Epoch = 38, LR = 0,000395, AvgCost = 198,7810, Sent = 757, SentPerMin = 135,37, WordPerSec = 124,52
Update = 23182, Epoch = 38, LR = 0,000394, AvgCost = 197,9072, Sent = 1000, SentPerMin = 135,43, WordPerSec = 124,29
Update = 23200, Epoch = 39, LR = 0,000394, AvgCost = 186,6355, Sent = 59, SentPerMin = 141,39, WordPerSec = 118,15
Update = 23300, Epoch = 39, LR = 0,000393, AvgCost = 202,5935, Sent = 343, SentPerMin = 139,24, WordPerSec = 128,74
Update = 23400, Epoch = 39, LR = 0,000392, AvgCost = 198,5031, Sent = 640, SentPerMin = 148,07, WordPerSec = 136,07
Update = 23500, Epoch = 39, LR = 0,000391, AvgCost = 197,4090, Sent = 943, SentPerMin = 151,71, WordPerSec = 139,25
Update = 23520, Epoch = 39, LR = 0,000391, AvgCost = 197,3887, Sent = 1000, SentPerMin = 152,72, WordPerSec = 140,16
Starting inference...
Inference results:
的•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••
的。
(
的。
的


             */

            /*
             Update = 10100, Epoch = 0, LR = 0,000597, AvgCost = 267,2337, Sent = 285, SentPerMin = 132,44, WordPerSec = 124,36
Update = 10200, Epoch = 0, LR = 0,000594, AvgCost = 248,1829, Sent = 592, SentPerMin = 132,41, WordPerSec = 121,46
Update = 10300, Epoch = 0, LR = 0,000591, AvgCost = 245,1342, Sent = 882, SentPerMin = 132,24, WordPerSec = 122,17
Update = 10338, Epoch = 0, LR = 0,000590, AvgCost = 241,9924, Sent = 1000, SentPerMin = 131,99, WordPerSec = 121,14
Update = 10400, Epoch = 1, LR = 0,000588, AvgCost = 232,4427, Sent = 178, SentPerMin = 157,06, WordPerSec = 144,93
Update = 10500, Epoch = 1, LR = 0,000586, AvgCost = 233,2338, Sent = 475, SentPerMin = 159,23, WordPerSec = 146,67
Update = 10600, Epoch = 1, LR = 0,000583, AvgCost = 231,8214, Sent = 771, SentPerMin = 159,60, WordPerSec = 147,51
Update = 10676, Epoch = 1, LR = 0,000581, AvgCost = 228,7093, Sent = 1000, SentPerMin = 156,16, WordPerSec = 143,31
Update = 10700, Epoch = 2, LR = 0,000580, AvgCost = 219,9052, Sent = 76, SentPerMin = 110,47, WordPerSec = 97,43
Update = 10800, Epoch = 2, LR = 0,000577, AvgCost = 230,2210, Sent = 363, SentPerMin = 113,13, WordPerSec = 104,32
Update = 10900, Epoch = 2, LR = 0,000575, AvgCost = 227,2912, Sent = 656, SentPerMin = 116,75, WordPerSec = 107,50
Update = 11000, Epoch = 2, LR = 0,000572, AvgCost = 224,7039, Sent = 961, SentPerMin = 116,95, WordPerSec = 107,19
Update = 11014, Epoch = 2, LR = 0,000572, AvgCost = 225,1164, Sent = 1000, SentPerMin = 117,29, WordPerSec = 107,64
Update = 11100, Epoch = 3, LR = 0,000569, AvgCost = 228,3545, Sent = 248, SentPerMin = 163,24, WordPerSec = 150,80
Update = 11200, Epoch = 3, LR = 0,000567, AvgCost = 224,3539, Sent = 549, SentPerMin = 165,01, WordPerSec = 151,19
Update = 11300, Epoch = 3, LR = 0,000564, AvgCost = 224,3099, Sent = 841, SentPerMin = 166,26, WordPerSec = 153,10
Update = 11352, Epoch = 3, LR = 0,000563, AvgCost = 222,8869, Sent = 1000, SentPerMin = 167,63, WordPerSec = 153,85
Update = 11400, Epoch = 4, LR = 0,000562, AvgCost = 220,8006, Sent = 140, SentPerMin = 139,07, WordPerSec = 124,83
Update = 11500, Epoch = 4, LR = 0,000560, AvgCost = 227,7481, Sent = 431, SentPerMin = 134,96, WordPerSec = 125,10
Update = 11600, Epoch = 4, LR = 0,000557, AvgCost = 220,7626, Sent = 732, SentPerMin = 137,34, WordPerSec = 125,57
Update = 11690, Epoch = 4, LR = 0,000555, AvgCost = 221,1459, Sent = 1000, SentPerMin = 137,29, WordPerSec = 126,00
Update = 11700, Epoch = 5, LR = 0,000555, AvgCost = 183,0179, Sent = 33, SentPerMin = 186,39, WordPerSec = 144,03
Update = 11800, Epoch = 5, LR = 0,000552, AvgCost = 227,5401, Sent = 317, SentPerMin = 166,52, WordPerSec = 155,07
Update = 11900, Epoch = 5, LR = 0,000550, AvgCost = 221,0203, Sent = 618, SentPerMin = 167,79, WordPerSec = 154,00
Update = 12000, Epoch = 5, LR = 0,000548, AvgCost = 220,6904, Sent = 914, SentPerMin = 168,33, WordPerSec = 155,12
Update = 12028, Epoch = 5, LR = 0,000547, AvgCost = 219,5172, Sent = 1000, SentPerMin = 169,41, WordPerSec = 155,48
Update = 12100, Epoch = 6, LR = 0,000545, AvgCost = 226,6129, Sent = 205, SentPerMin = 135,43, WordPerSec = 126,45
Update = 12200, Epoch = 6, LR = 0,000543, AvgCost = 219,6765, Sent = 506, SentPerMin = 135,67, WordPerSec = 124,44
Update = 12300, Epoch = 6, LR = 0,000541, AvgCost = 219,6219, Sent = 802, SentPerMin = 136,24, WordPerSec = 125,43
Update = 12366, Epoch = 6, LR = 0,000540, AvgCost = 217,7773, Sent = 1000, SentPerMin = 137,41, WordPerSec = 126,11
Update = 12400, Epoch = 7, LR = 0,000539, AvgCost = 207,8832, Sent = 106, SentPerMin = 171,98, WordPerSec = 148,13
Update = 12500, Epoch = 7, LR = 0,000537, AvgCost = 220,3620, Sent = 393, SentPerMin = 165,57, WordPerSec = 152,60
Update = 12600, Epoch = 7, LR = 0,000535, AvgCost = 216,8614, Sent = 689, SentPerMin = 169,48, WordPerSec = 155,36
Update = 12700, Epoch = 7, LR = 0,000532, AvgCost = 216,4828, Sent = 989, SentPerMin = 169,35, WordPerSec = 155,47
Update = 12704, Epoch = 7, LR = 0,000532, AvgCost = 216,3343, Sent = 1000, SentPerMin = 169,48, WordPerSec = 155,54
Update = 12800, Epoch = 8, LR = 0,000530, AvgCost = 226,8765, Sent = 275, SentPerMin = 134,69, WordPerSec = 126,44
Update = 12900, Epoch = 8, LR = 0,000528, AvgCost = 216,5665, Sent = 581, SentPerMin = 134,13, WordPerSec = 123,05
Update = 13000, Epoch = 8, LR = 0,000526, AvgCost = 216,8050, Sent = 870, SentPerMin = 135,41, WordPerSec = 125,02
Update = 13042, Epoch = 8, LR = 0,000525, AvgCost = 215,3095, Sent = 1000, SentPerMin = 136,47, WordPerSec = 125,25
Update = 13100, Epoch = 9, LR = 0,000524, AvgCost = 218,3259, Sent = 166, SentPerMin = 166,60, WordPerSec = 153,52
Update = 13200, Epoch = 9, LR = 0,000522, AvgCost = 218,1949, Sent = 463, SentPerMin = 165,79, WordPerSec = 153,03
Update = 13300, Epoch = 9, LR = 0,000520, AvgCost = 215,9962, Sent = 761, SentPerMin = 166,74, WordPerSec = 153,54
Update = 13380, Epoch = 9, LR = 0,000519, AvgCost = 213,8852, Sent = 1000, SentPerMin = 169,11, WordPerSec = 155,20
Update = 13400, Epoch = 10, LR = 0,000518, AvgCost = 209,6116, Sent = 64, SentPerMin = 140,13, WordPerSec = 122,17
Update = 13500, Epoch = 10, LR = 0,000516, AvgCost = 217,6910, Sent = 352, SentPerMin = 135,27, WordPerSec = 124,40
Update = 13600, Epoch = 10, LR = 0,000514, AvgCost = 214,9194, Sent = 645, SentPerMin = 135,93, WordPerSec = 125,11
Update = 13700, Epoch = 10, LR = 0,000513, AvgCost = 214,0757, Sent = 948, SentPerMin = 136,63, WordPerSec = 125,61
Update = 13718, Epoch = 10, LR = 0,000512, AvgCost = 213,6422, Sent = 1000, SentPerMin = 137,19, WordPerSec = 125,91
Update = 13800, Epoch = 11, LR = 0,000511, AvgCost = 215,2252, Sent = 239, SentPerMin = 170,59, WordPerSec = 155,79
Update = 13900, Epoch = 11, LR = 0,000509, AvgCost = 214,9297, Sent = 536, SentPerMin = 166,92, WordPerSec = 153,43
Update = 14000, Epoch = 11, LR = 0,000507, AvgCost = 213,6787, Sent = 831, SentPerMin = 168,04, WordPerSec = 154,30
Update = 14056, Epoch = 11, LR = 0,000506, AvgCost = 212,9276, Sent = 1000, SentPerMin = 168,95, WordPerSec = 155,05
Update = 14100, Epoch = 12, LR = 0,000505, AvgCost = 211,5359, Sent = 131, SentPerMin = 139,62, WordPerSec = 124,25
Update = 14200, Epoch = 12, LR = 0,000504, AvgCost = 215,2752, Sent = 423, SentPerMin = 135,31, WordPerSec = 124,35
Update = 14300, Epoch = 12, LR = 0,000502, AvgCost = 212,4001, Sent = 720, SentPerMin = 137,20, WordPerSec = 125,56
Update = 14394, Epoch = 12, LR = 0,000500, AvgCost = 212,6715, Sent = 1000, SentPerMin = 137,25, WordPerSec = 125,96
Update = 14400, Epoch = 13, LR = 0,000500, AvgCost = 211,1852, Sent = 16, SentPerMin = 164,13, WordPerSec = 160,20
Update = 14500, Epoch = 13, LR = 0,000498, AvgCost = 219,6845, Sent = 304, SentPerMin = 167,05, WordPerSec = 156,02
Update = 14600, Epoch = 13, LR = 0,000497, AvgCost = 213,0419, Sent = 606, SentPerMin = 167,20, WordPerSec = 153,67
Update = 14700, Epoch = 13, LR = 0,000495, AvgCost = 212,7473, Sent = 902, SentPerMin = 167,99, WordPerSec = 154,71
Update = 14732, Epoch = 13, LR = 0,000494, AvgCost = 211,5317, Sent = 1000, SentPerMin = 169,02, WordPerSec = 155,12
Update = 14800, Epoch = 14, LR = 0,000493, AvgCost = 216,7171, Sent = 195, SentPerMin = 136,88, WordPerSec = 125,81
Update = 14900, Epoch = 14, LR = 0,000492, AvgCost = 212,5137, Sent = 495, SentPerMin = 136,50, WordPerSec = 124,75
Update = 15000, Epoch = 14, LR = 0,000490, AvgCost = 213,3280, Sent = 789, SentPerMin = 135,72, WordPerSec = 125,14
Update = 15070, Epoch = 14, LR = 0,000489, AvgCost = 211,0268, Sent = 1000, SentPerMin = 137,13, WordPerSec = 125,86
Update = 15100, Epoch = 15, LR = 0,000488, AvgCost = 208,4882, Sent = 92, SentPerMin = 166,12, WordPerSec = 147,43
Update = 15200, Epoch = 15, LR = 0,000487, AvgCost = 213,7355, Sent = 381, SentPerMin = 165,36, WordPerSec = 152,16
Update = 15300, Epoch = 15, LR = 0,000485, AvgCost = 211,3946, Sent = 676, SentPerMin = 167,90, WordPerSec = 154,14
Update = 15400, Epoch = 15, LR = 0,000483, AvgCost = 209,1041, Sent = 979, SentPerMin = 168,93, WordPerSec = 154,68
Update = 15408, Epoch = 15, LR = 0,000483, AvgCost = 210,1424, Sent = 1000, SentPerMin = 168,72, WordPerSec = 154,85
Update = 15500, Epoch = 16, LR = 0,000482, AvgCost = 217,1947, Sent = 265, SentPerMin = 136,38, WordPerSec = 126,57
Update = 15600, Epoch = 16, LR = 0,000480, AvgCost = 211,4123, Sent = 566, SentPerMin = 135,74, WordPerSec = 124,93
Update = 15700, Epoch = 16, LR = 0,000479, AvgCost = 210,9266, Sent = 859, SentPerMin = 136,12, WordPerSec = 125,44
Update = 15746, Epoch = 16, LR = 0,000478, AvgCost = 209,6252, Sent = 1000, SentPerMin = 137,20, WordPerSec = 125,91
Update = 15800, Epoch = 17, LR = 0,000477, AvgCost = 211,5984, Sent = 157, SentPerMin = 167,56, WordPerSec = 152,42
Update = 15900, Epoch = 17, LR = 0,000476, AvgCost = 213,0831, Sent = 450, SentPerMin = 166,23, WordPerSec = 153,29
Update = 16000, Epoch = 17, LR = 0,000474, AvgCost = 208,6958, Sent = 752, SentPerMin = 167,25, WordPerSec = 153,34
Update = 16084, Epoch = 17, LR = 0,000473, AvgCost = 208,7335, Sent = 1000, SentPerMin = 169,00, WordPerSec = 155,10
Update = 16100, Epoch = 18, LR = 0,000473, AvgCost = 201,9302, Sent = 51, SentPerMin = 142,95, WordPerSec = 123,24
Update = 16200, Epoch = 18, LR = 0,000471, AvgCost = 214,3337, Sent = 336, SentPerMin = 135,52, WordPerSec = 125,56
Update = 16300, Epoch = 18, LR = 0,000470, AvgCost = 209,1384, Sent = 636, SentPerMin = 136,06, WordPerSec = 124,66
Update = 16400, Epoch = 18, LR = 0,000469, AvgCost = 209,1350, Sent = 935, SentPerMin = 136,41, WordPerSec = 125,39
Update = 16422, Epoch = 18, LR = 0,000468, AvgCost = 208,6610, Sent = 1000, SentPerMin = 137,18, WordPerSec = 125,90
Update = 16500, Epoch = 19, LR = 0,000467, AvgCost = 210,7391, Sent = 225, SentPerMin = 169,86, WordPerSec = 156,14
Update = 16600, Epoch = 19, LR = 0,000466, AvgCost = 210,0801, Sent = 524, SentPerMin = 166,62, WordPerSec = 153,32
Update = 16700, Epoch = 19, LR = 0,000464, AvgCost = 208,7056, Sent = 819, SentPerMin = 167,66, WordPerSec = 154,21
Update = 16760, Epoch = 19, LR = 0,000463, AvgCost = 207,5485, Sent = 1000, SentPerMin = 168,91, WordPerSec = 155,02
Update = 16800, Epoch = 20, LR = 0,000463, AvgCost = 203,2285, Sent = 122, SentPerMin = 140,92, WordPerSec = 123,00
Update = 16900, Epoch = 20, LR = 0,000462, AvgCost = 210,5576, Sent = 411, SentPerMin = 134,74, WordPerSec = 124,10
Update = 17000, Epoch = 20, LR = 0,000460, AvgCost = 206,9975, Sent = 709, SentPerMin = 137,56, WordPerSec = 125,65
Update = 17098, Epoch = 20, LR = 0,000459, AvgCost = 207,1652, Sent = 1000, SentPerMin = 137,14, WordPerSec = 125,86
Update = 17100, Epoch = 21, LR = 0,000459, AvgCost = 183,4794, Sent = 6, SentPerMin = 183,76, WordPerSec = 166,92
Update = 17200, Epoch = 21, LR = 0,000457, AvgCost = 215,4899, Sent = 292, SentPerMin = 166,50, WordPerSec = 155,78
Update = 17300, Epoch = 21, LR = 0,000456, AvgCost = 208,6516, Sent = 595, SentPerMin = 166,73, WordPerSec = 153,46
Update = 17400, Epoch = 21, LR = 0,000455, AvgCost = 207,6194, Sent = 890, SentPerMin = 167,85, WordPerSec = 154,72
Update = 17436, Epoch = 21, LR = 0,000454, AvgCost = 206,1395, Sent = 1000, SentPerMin = 169,02, WordPerSec = 155,12
Update = 17500, Epoch = 22, LR = 0,000454, AvgCost = 212,8126, Sent = 184, SentPerMin = 136,69, WordPerSec = 125,59
Update = 17600, Epoch = 22, LR = 0,000452, AvgCost = 209,3183, Sent = 482, SentPerMin = 135,56, WordPerSec = 124,63
Update = 17700, Epoch = 22, LR = 0,000451, AvgCost = 209,0892, Sent = 775, SentPerMin = 134,88, WordPerSec = 124,97
Update = 17774, Epoch = 22, LR = 0,000450, AvgCost = 205,8094, Sent = 1000, SentPerMin = 137,00, WordPerSec = 125,73
Update = 17800, Epoch = 23, LR = 0,000450, AvgCost = 199,8052, Sent = 81, SentPerMin = 165,50, WordPerSec = 146,91
Update = 17900, Epoch = 23, LR = 0,000448, AvgCost = 210,7833, Sent = 368, SentPerMin = 164,10, WordPerSec = 151,86
Update = 18000, Epoch = 23, LR = 0,000447, AvgCost = 206,5254, Sent = 663, SentPerMin = 167,23, WordPerSec = 153,65
Update = 18100, Epoch = 23, LR = 0,000446, AvgCost = 203,9870, Sent = 969, SentPerMin = 168,49, WordPerSec = 154,07
Update = 18112, Epoch = 23, LR = 0,000446, AvgCost = 204,8513, Sent = 1000, SentPerMin = 168,39, WordPerSec = 154,54
Update = 18200, Epoch = 24, LR = 0,000445, AvgCost = 215,1501, Sent = 251, SentPerMin = 135,59, WordPerSec = 126,66
Update = 18300, Epoch = 24, LR = 0,000444, AvgCost = 206,1311, Sent = 556, SentPerMin = 136,41, WordPerSec = 124,86
Update = 18400, Epoch = 24, LR = 0,000442, AvgCost = 206,1314, Sent = 847, SentPerMin = 136,10, WordPerSec = 125,45
Update = 18450, Epoch = 24, LR = 0,000442, AvgCost = 204,6779, Sent = 1000, SentPerMin = 137,02, WordPerSec = 125,75
Update = 18500, Epoch = 25, LR = 0,000441, AvgCost = 206,5935, Sent = 145, SentPerMin = 166,68, WordPerSec = 151,60
Update = 18600, Epoch = 25, LR = 0,000440, AvgCost = 209,9102, Sent = 437, SentPerMin = 165,54, WordPerSec = 153,41
Update = 18700, Epoch = 25, LR = 0,000439, AvgCost = 203,8061, Sent = 739, SentPerMin = 167,12, WordPerSec = 153,20
Update = 18788, Epoch = 25, LR = 0,000438, AvgCost = 203,6519, Sent = 1000, SentPerMin = 168,51, WordPerSec = 154,66
Update = 18800, Epoch = 26, LR = 0,000438, AvgCost = 165,0030, Sent = 41, SentPerMin = 149,99, WordPerSec = 115,48
Update = 18900, Epoch = 26, LR = 0,000436, AvgCost = 210,9336, Sent = 324, SentPerMin = 135,02, WordPerSec = 125,42
Update = 19000, Epoch = 26, LR = 0,000435, AvgCost = 204,9388, Sent = 624, SentPerMin = 135,68, WordPerSec = 124,39
Update = 19100, Epoch = 26, LR = 0,000434, AvgCost = 204,5033, Sent = 921, SentPerMin = 136,39, WordPerSec = 125,48
Update = 19126, Epoch = 26, LR = 0,000434, AvgCost = 203,6528, Sent = 1000, SentPerMin = 136,84, WordPerSec = 125,59
Update = 19200, Epoch = 27, LR = 0,000433, AvgCost = 208,6458, Sent = 213, SentPerMin = 167,97, WordPerSec = 154,96
Update = 19300, Epoch = 27, LR = 0,000432, AvgCost = 206,2857, Sent = 511, SentPerMin = 165,38, WordPerSec = 152,50
Update = 19400, Epoch = 27, LR = 0,000431, AvgCost = 204,4202, Sent = 808, SentPerMin = 167,37, WordPerSec = 154,17
Update = 19464, Epoch = 27, LR = 0,000430, AvgCost = 202,6176, Sent = 1000, SentPerMin = 168,81, WordPerSec = 154,93
Update = 19500, Epoch = 28, LR = 0,000430, AvgCost = 191,5741, Sent = 113, SentPerMin = 141,27, WordPerSec = 121,02
Update = 19600, Epoch = 28, LR = 0,000429, AvgCost = 205,6108, Sent = 399, SentPerMin = 134,26, WordPerSec = 123,58
Update = 19700, Epoch = 28, LR = 0,000427, AvgCost = 202,9052, Sent = 696, SentPerMin = 136,40, WordPerSec = 124,97
Update = 19800, Epoch = 28, LR = 0,000426, AvgCost = 202,6074, Sent = 995, SentPerMin = 136,71, WordPerSec = 125,38
Update = 19802, Epoch = 28, LR = 0,000426, AvgCost = 202,5550, Sent = 1000, SentPerMin = 136,66, WordPerSec = 125,42
Update = 19900, Epoch = 29, LR = 0,000425, AvgCost = 211,6482, Sent = 280, SentPerMin = 164,36, WordPerSec = 154,22
Update = 20000, Epoch = 29, LR = 0,000424, AvgCost = 201,8881, Sent = 588, SentPerMin = 165,77, WordPerSec = 151,64
Update = 20100, Epoch = 29, LR = 0,000423, AvgCost = 202,5445, Sent = 877, SentPerMin = 128,20, WordPerSec = 118,23
Update = 20140, Epoch = 29, LR = 0,000423, AvgCost = 201,3814, Sent = 1000, SentPerMin = 132,87, WordPerSec = 121,94
Update = 20200, Epoch = 30, LR = 0,000422, AvgCost = 204,1076, Sent = 173, SentPerMin = 162,16, WordPerSec = 149,19
Update = 20300, Epoch = 30, LR = 0,000421, AvgCost = 205,1951, Sent = 469, SentPerMin = 140,65, WordPerSec = 129,73
Update = 20400, Epoch = 30, LR = 0,000420, AvgCost = 203,5834, Sent = 766, SentPerMin = 137,18, WordPerSec = 126,58
Update = 20478, Epoch = 30, LR = 0,000419, AvgCost = 201,3124, Sent = 1000, SentPerMin = 137,07, WordPerSec = 125,80
Update = 20500, Epoch = 31, LR = 0,000419, AvgCost = 194,2038, Sent = 71, SentPerMin = 132,54, WordPerSec = 114,71
Update = 20600, Epoch = 31, LR = 0,000418, AvgCost = 204,4081, Sent = 358, SentPerMin = 143,52, WordPerSec = 131,84
Update = 20700, Epoch = 31, LR = 0,000417, AvgCost = 201,4487, Sent = 652, SentPerMin = 151,04, WordPerSec = 138,70
Update = 20800, Epoch = 31, LR = 0,000416, AvgCost = 200,2266, Sent = 955, SentPerMin = 154,35, WordPerSec = 141,61
Update = 20816, Epoch = 31, LR = 0,000416, AvgCost = 200,2224, Sent = 1000, SentPerMin = 155,01, WordPerSec = 142,26
Update = 20900, Epoch = 32, LR = 0,000415, AvgCost = 202,0987, Sent = 244, SentPerMin = 150,07, WordPerSec = 137,10
Update = 21000, Epoch = 32, LR = 0,000414, AvgCost = 201,4620, Sent = 542, SentPerMin = 136,15, WordPerSec = 125,04
Update = 21100, Epoch = 32, LR = 0,000413, AvgCost = 201,1664, Sent = 835, SentPerMin = 134,07, WordPerSec = 123,35
Update = 21154, Epoch = 32, LR = 0,000413, AvgCost = 200,0226, Sent = 1000, SentPerMin = 134,15, WordPerSec = 123,12
Update = 21200, Epoch = 33, LR = 0,000412, AvgCost = 197,7788, Sent = 136, SentPerMin = 132,91, WordPerSec = 118,27
Update = 21300, Epoch = 33, LR = 0,000411, AvgCost = 201,8704, Sent = 428, SentPerMin = 147,10, WordPerSec = 135,30
Update = 21400, Epoch = 33, LR = 0,000410, AvgCost = 199,3914, Sent = 725, SentPerMin = 154,35, WordPerSec = 141,49
Update = 21492, Epoch = 33, LR = 0,000409, AvgCost = 199,1107, Sent = 1000, SentPerMin = 157,06, WordPerSec = 144,15
Update = 21500, Epoch = 34, LR = 0,000409, AvgCost = 181,9554, Sent = 23, SentPerMin = 162,19, WordPerSec = 141,39
Update = 21600, Epoch = 34, LR = 0,000408, AvgCost = 205,8990, Sent = 311, SentPerMin = 144,26, WordPerSec = 134,13
Update = 21700, Epoch = 34, LR = 0,000407, AvgCost = 200,8706, Sent = 611, SentPerMin = 137,28, WordPerSec = 126,44
Update = 21800, Epoch = 34, LR = 0,000406, AvgCost = 200,4808, Sent = 906, SentPerMin = 136,35, WordPerSec = 125,78
Update = 21830, Epoch = 34, LR = 0,000406, AvgCost = 198,9705, Sent = 1000, SentPerMin = 136,78, WordPerSec = 125,54
Update = 21900, Epoch = 35, LR = 0,000405, AvgCost = 202,7799, Sent = 200, SentPerMin = 138,31, WordPerSec = 126,92
Update = 22000, Epoch = 35, LR = 0,000405, AvgCost = 199,2817, Sent = 501, SentPerMin = 151,23, WordPerSec = 138,55
Update = 22100, Epoch = 35, LR = 0,000404, AvgCost = 199,8779, Sent = 795, SentPerMin = 155,41, WordPerSec = 143,19
Update = 22168, Epoch = 35, LR = 0,000403, AvgCost = 197,8971, Sent = 1000, SentPerMin = 158,38, WordPerSec = 145,36
Update = 22200, Epoch = 36, LR = 0,000403, AvgCost = 192,3833, Sent = 99, SentPerMin = 166,40, WordPerSec = 145,00
Update = 22300, Epoch = 36, LR = 0,000402, AvgCost = 200,0044, Sent = 388, SentPerMin = 142,55, WordPerSec = 130,65
Update = 22400, Epoch = 36, LR = 0,000401, AvgCost = 198,4066, Sent = 683, SentPerMin = 138,61, WordPerSec = 127,01
Update = 22500, Epoch = 36, LR = 0,000400, AvgCost = 198,1460, Sent = 984, SentPerMin = 136,71, WordPerSec = 125,46
Update = 22506, Epoch = 36, LR = 0,000400, AvgCost = 198,0110, Sent = 1000, SentPerMin = 136,72, WordPerSec = 125,48
Update = 22600, Epoch = 37, LR = 0,000399, AvgCost = 205,8093, Sent = 269, SentPerMin = 142,52, WordPerSec = 133,21
Update = 22700, Epoch = 37, LR = 0,000398, AvgCost = 198,4249, Sent = 573, SentPerMin = 152,18, WordPerSec = 139,77
Update = 22800, Epoch = 37, LR = 0,000397, AvgCost = 198,0576, Sent = 865, SentPerMin = 155,74, WordPerSec = 143,45
Update = 22844, Epoch = 37, LR = 0,000397, AvgCost = 197,1488, Sent = 1000, SentPerMin = 157,71, WordPerSec = 144,74
Update = 22900, Epoch = 38, LR = 0,000396, AvgCost = 200,0437, Sent = 161, SentPerMin = 160,24, WordPerSec = 146,98
Update = 23000, Epoch = 38, LR = 0,000396, AvgCost = 201,0646, Sent = 456, SentPerMin = 138,88, WordPerSec = 128,32
Update = 23100, Epoch = 38, LR = 0,000395, AvgCost = 198,2510, Sent = 757, SentPerMin = 136,24, WordPerSec = 125,31
Update = 23182, Epoch = 38, LR = 0,000394, AvgCost = 197,3228, Sent = 1000, SentPerMin = 136,81, WordPerSec = 125,56
Update = 23200, Epoch = 39, LR = 0,000394, AvgCost = 185,2258, Sent = 59, SentPerMin = 143,50, WordPerSec = 119,91
Update = 23300, Epoch = 39, LR = 0,000393, AvgCost = 201,5196, Sent = 343, SentPerMin = 146,30, WordPerSec = 135,27
Update = 23400, Epoch = 39, LR = 0,000392, AvgCost = 197,4091, Sent = 640, SentPerMin = 153,99, WordPerSec = 141,51
Update = 23500, Epoch = 39, LR = 0,000391, AvgCost = 196,5991, Sent = 943, SentPerMin = 157,34, WordPerSec = 144,42
Update = 23520, Epoch = 39, LR = 0,000391, AvgCost = 196,4445, Sent = 1000, SentPerMin = 158,40, WordPerSec = 145,37
Update = 23600, Epoch = 40, LR = 0,000391, AvgCost = 199,2331, Sent = 231, SentPerMin = 154,23, WordPerSec = 141,92
Update = 23700, Epoch = 40, LR = 0,000390, AvgCost = 198,3504, Sent = 529, SentPerMin = 139,22, WordPerSec = 128,14
Update = 23800, Epoch = 40, LR = 0,000389, AvgCost = 197,3549, Sent = 824, SentPerMin = 137,13, WordPerSec = 126,18
Update = 23858, Epoch = 40, LR = 0,000388, AvgCost = 196,2306, Sent = 1000, SentPerMin = 137,16, WordPerSec = 125,88
Update = 23900, Epoch = 41, LR = 0,000388, AvgCost = 193,1778, Sent = 126, SentPerMin = 135,80, WordPerSec = 119,88
Update = 24000, Epoch = 41, LR = 0,000387, AvgCost = 197,6080, Sent = 418, SentPerMin = 148,59, WordPerSec = 136,35
Update = 24100, Epoch = 41, LR = 0,000386, AvgCost = 195,0972, Sent = 715, SentPerMin = 156,16, WordPerSec = 142,80
Update = 24196, Epoch = 41, LR = 0,000386, AvgCost = 195,4197, Sent = 1000, SentPerMin = 158,40, WordPerSec = 145,37
Update = 24200, Epoch = 42, LR = 0,000386, AvgCost = 173,8035, Sent = 12, SentPerMin = 181,57, WordPerSec = 163,41
Update = 24300, Epoch = 42, LR = 0,000385, AvgCost = 203,3988, Sent = 299, SentPerMin = 144,71, WordPerSec = 134,99
Update = 24400, Epoch = 42, LR = 0,000384, AvgCost = 196,6818, Sent = 601, SentPerMin = 137,37, WordPerSec = 126,19
Update = 24500, Epoch = 42, LR = 0,000383, AvgCost = 196,1621, Sent = 897, SentPerMin = 136,32, WordPerSec = 125,39
Update = 24534, Epoch = 42, LR = 0,000383, AvgCost = 195,2699, Sent = 1000, SentPerMin = 136,45, WordPerSec = 125,23
Update = 24600, Epoch = 43, LR = 0,000383, AvgCost = 201,0835, Sent = 189, SentPerMin = 135,62, WordPerSec = 124,93
Update = 24700, Epoch = 43, LR = 0,000382, AvgCost = 196,2260, Sent = 490, SentPerMin = 150,57, WordPerSec = 137,65
Update = 24800, Epoch = 43, LR = 0,000381, AvgCost = 196,7567, Sent = 784, SentPerMin = 153,97, WordPerSec = 141,90
Update = 24872, Epoch = 43, LR = 0,000380, AvgCost = 194,5363, Sent = 1000, SentPerMin = 157,43, WordPerSec = 144,48
Update = 24900, Epoch = 44, LR = 0,000380, AvgCost = 188,6153, Sent = 87, SentPerMin = 164,29, WordPerSec = 144,37
Update = 25000, Epoch = 44, LR = 0,000379, AvgCost = 198,5021, Sent = 375, SentPerMin = 140,71, WordPerSec = 129,69
Update = 25100, Epoch = 44, LR = 0,000379, AvgCost = 196,3810, Sent = 668, SentPerMin = 137,02, WordPerSec = 126,26
Update = 25200, Epoch = 44, LR = 0,000378, AvgCost = 193,8929, Sent = 974, SentPerMin = 136,40, WordPerSec = 124,82
Update = 25210, Epoch = 44, LR = 0,000378, AvgCost = 194,8136, Sent = 1000, SentPerMin = 136,16, WordPerSec = 124,97
Update = 25300, Epoch = 45, LR = 0,000377, AvgCost = 202,8559, Sent = 258, SentPerMin = 141,79, WordPerSec = 132,17
Update = 25400, Epoch = 45, LR = 0,000376, AvgCost = 195,1365, Sent = 562, SentPerMin = 152,42, WordPerSec = 139,76
Update = 25500, Epoch = 45, LR = 0,000376, AvgCost = 195,4401, Sent = 854, SentPerMin = 155,85, WordPerSec = 143,52
Update = 25548, Epoch = 45, LR = 0,000375, AvgCost = 194,1764, Sent = 1000, SentPerMin = 157,97, WordPerSec = 144,98
Update = 25600, Epoch = 46, LR = 0,000375, AvgCost = 193,0467, Sent = 152, SentPerMin = 162,62, WordPerSec = 146,31
Update = 25700, Epoch = 46, LR = 0,000374, AvgCost = 198,6939, Sent = 443, SentPerMin = 139,54, WordPerSec = 129,02
Update = 25800, Epoch = 46, LR = 0,000374, AvgCost = 194,6217, Sent = 744, SentPerMin = 136,57, WordPerSec = 125,43
Update = 25886, Epoch = 46, LR = 0,000373, AvgCost = 194,3149, Sent = 1000, SentPerMin = 136,35, WordPerSec = 125,13
Update = 25900, Epoch = 47, LR = 0,000373, AvgCost = 181,7918, Sent = 45, SentPerMin = 139,35, WordPerSec = 114,11
Update = 26000, Epoch = 47, LR = 0,000372, AvgCost = 199,9160, Sent = 330, SentPerMin = 145,25, WordPerSec = 134,64
Update = 26100, Epoch = 47, LR = 0,000371, AvgCost = 194,8065, Sent = 630, SentPerMin = 152,91, WordPerSec = 140,22
Update = 26200, Epoch = 47, LR = 0,000371, AvgCost = 194,1806, Sent = 929, SentPerMin = 156,25, WordPerSec = 143,55
Update = 26224, Epoch = 47, LR = 0,000371, AvgCost = 193,6445, Sent = 1000, SentPerMin = 157,51, WordPerSec = 144,56
Update = 26300, Epoch = 48, LR = 0,000370, AvgCost = 197,2726, Sent = 219, SentPerMin = 152,52, WordPerSec = 140,54
Update = 26400, Epoch = 48, LR = 0,000369, AvgCost = 195,8017, Sent = 519, SentPerMin = 137,94, WordPerSec = 126,57
Update = 26500, Epoch = 48, LR = 0,000369, AvgCost = 195,0285, Sent = 814, SentPerMin = 135,85, WordPerSec = 124,96
Update = 26562, Epoch = 48, LR = 0,000368, AvgCost = 193,7387, Sent = 1000, SentPerMin = 135,84, WordPerSec = 124,67
Update = 26600, Epoch = 49, LR = 0,000368, AvgCost = 179,8796, Sent = 119, SentPerMin = 137,55, WordPerSec = 117,02
Update = 26700, Epoch = 49, LR = 0,000367, AvgCost = 195,8604, Sent = 406, SentPerMin = 146,56, WordPerSec = 134,70
Update = 26800, Epoch = 49, LR = 0,000367, AvgCost = 193,2162, Sent = 702, SentPerMin = 154,28, WordPerSec = 141,09
Update = 26900, Epoch = 49, LR = 0,000366, AvgCost = 193,1741, Sent = 1000, SentPerMin = 156,67, WordPerSec = 143,79
Update = 26900, Epoch = 49, LR = 0,000366, AvgCost = 193,1741, Sent = 1000, SentPerMin = 156,67, WordPerSec = 143,79
Starting inference...
Inference results:
在在在在在在在在在在在在在在在在。""""""
。
在在在在在在在。    的在。
在在
在。


             */

            /*
             Update = 10100, Epoch = 0, LR = 0,000597, AvgCost = 266,7051, Sent = 285, SentPerMin = 819,27, WordPerSec = 769,30
Update = 10200, Epoch = 0, LR = 0,000594, AvgCost = 248,1315, Sent = 592, SentPerMin = 865,67, WordPerSec = 794,09
Update = 10300, Epoch = 0, LR = 0,000591, AvgCost = 245,1942, Sent = 882, SentPerMin = 867,29, WordPerSec = 801,22
Update = 10338, Epoch = 0, LR = 0,000590, AvgCost = 242,0906, Sent = 1000, SentPerMin = 875,47, WordPerSec = 803,48
Update = 10400, Epoch = 1, LR = 0,000588, AvgCost = 232,6859, Sent = 178, SentPerMin = 1079,58, WordPerSec = 996,19
Update = 10500, Epoch = 1, LR = 0,000586, AvgCost = 233,5576, Sent = 475, SentPerMin = 1082,00, WordPerSec = 996,65
Update = 10600, Epoch = 1, LR = 0,000583, AvgCost = 232,1025, Sent = 771, SentPerMin = 1083,46, WordPerSec = 1001,44
Update = 10676, Epoch = 1, LR = 0,000581, AvgCost = 228,9344, Sent = 1000, SentPerMin = 1095,11, WordPerSec = 1005,05
Update = 10700, Epoch = 2, LR = 0,000580, AvgCost = 220,2078, Sent = 76, SentPerMin = 1093,79, WordPerSec = 964,74
Update = 10800, Epoch = 2, LR = 0,000577, AvgCost = 230,5521, Sent = 363, SentPerMin = 1082,50, WordPerSec = 998,21
Update = 10900, Epoch = 2, LR = 0,000575, AvgCost = 227,3732, Sent = 656, SentPerMin = 1091,24, WordPerSec = 1004,74
Update = 11000, Epoch = 2, LR = 0,000572, AvgCost = 224,8955, Sent = 961, SentPerMin = 1107,87, WordPerSec = 1015,43
Update = 11014, Epoch = 2, LR = 0,000572, AvgCost = 225,3039, Sent = 1000, SentPerMin = 1108,15, WordPerSec = 1017,02
Update = 11100, Epoch = 3, LR = 0,000569, AvgCost = 228,3337, Sent = 248, SentPerMin = 1095,44, WordPerSec = 1011,96
Update = 11200, Epoch = 3, LR = 0,000567, AvgCost = 224,4920, Sent = 549, SentPerMin = 1111,71, WordPerSec = 1018,60
Update = 11300, Epoch = 3, LR = 0,000564, AvgCost = 224,4578, Sent = 841, SentPerMin = 1110,62, WordPerSec = 1022,71
Update = 11352, Epoch = 3, LR = 0,000563, AvgCost = 223,0399, Sent = 1000, SentPerMin = 1118,00, WordPerSec = 1026,06
Update = 11400, Epoch = 4, LR = 0,000562, AvgCost = 221,1547, Sent = 140, SentPerMin = 1113,99, WordPerSec = 999,94
Update = 11500, Epoch = 4, LR = 0,000560, AvgCost = 228,1384, Sent = 431, SentPerMin = 1095,13, WordPerSec = 1015,14
Update = 11600, Epoch = 4, LR = 0,000557, AvgCost = 221,1655, Sent = 732, SentPerMin = 1116,02, WordPerSec = 1020,32
Update = 11690, Epoch = 4, LR = 0,000555, AvgCost = 221,4133, Sent = 1000, SentPerMin = 1118,19, WordPerSec = 1026,24
Update = 11700, Epoch = 5, LR = 0,000555, AvgCost = 182,9030, Sent = 33, SentPerMin = 1222,56, WordPerSec = 944,70
Update = 11800, Epoch = 5, LR = 0,000552, AvgCost = 228,2750, Sent = 317, SentPerMin = 1082,14, WordPerSec = 1007,72
Update = 11900, Epoch = 5, LR = 0,000550, AvgCost = 222,0331, Sent = 618, SentPerMin = 1101,90, WordPerSec = 1011,29
Update = 12000, Epoch = 5, LR = 0,000548, AvgCost = 221,5040, Sent = 914, SentPerMin = 1106,64, WordPerSec = 1019,77
Update = 12028, Epoch = 5, LR = 0,000547, AvgCost = 220,2879, Sent = 1000, SentPerMin = 1112,43, WordPerSec = 1020,95
Update = 12100, Epoch = 6, LR = 0,000545, AvgCost = 228,1860, Sent = 205, SentPerMin = 1076,79, WordPerSec = 1005,35
Update = 12200, Epoch = 6, LR = 0,000543, AvgCost = 221,5841, Sent = 506, SentPerMin = 1100,30, WordPerSec = 1009,22
Update = 12300, Epoch = 6, LR = 0,000541, AvgCost = 221,3585, Sent = 802, SentPerMin = 1105,53, WordPerSec = 1017,79
Update = 12366, Epoch = 6, LR = 0,000540, AvgCost = 219,2984, Sent = 1000, SentPerMin = 1115,53, WordPerSec = 1023,80
Update = 12400, Epoch = 7, LR = 0,000539, AvgCost = 208,6609, Sent = 106, SentPerMin = 1148,82, WordPerSec = 989,50
Update = 12500, Epoch = 7, LR = 0,000537, AvgCost = 222,2520, Sent = 393, SentPerMin = 1098,79, WordPerSec = 1012,68
Update = 12600, Epoch = 7, LR = 0,000535, AvgCost = 218,8304, Sent = 689, SentPerMin = 1113,28, WordPerSec = 1020,50
Update = 12700, Epoch = 7, LR = 0,000532, AvgCost = 218,3753, Sent = 989, SentPerMin = 1113,95, WordPerSec = 1022,62
Update = 12704, Epoch = 7, LR = 0,000532, AvgCost = 218,2070, Sent = 1000, SentPerMin = 1113,96, WordPerSec = 1022,35
Update = 12800, Epoch = 8, LR = 0,000530, AvgCost = 228,3673, Sent = 275, SentPerMin = 1069,77, WordPerSec = 1004,22
Update = 12900, Epoch = 8, LR = 0,000528, AvgCost = 218,7738, Sent = 581, SentPerMin = 1098,05, WordPerSec = 1007,31
Update = 13000, Epoch = 8, LR = 0,000526, AvgCost = 219,0085, Sent = 870, SentPerMin = 1098,21, WordPerSec = 1013,95
Update = 13042, Epoch = 8, LR = 0,000525, AvgCost = 217,4876, Sent = 1000, SentPerMin = 1106,31, WordPerSec = 1015,33
Update = 13100, Epoch = 9, LR = 0,000524, AvgCost = 220,1978, Sent = 166, SentPerMin = 1075,88, WordPerSec = 991,41
Update = 13200, Epoch = 9, LR = 0,000522, AvgCost = 220,7729, Sent = 463, SentPerMin = 1088,06, WordPerSec = 1004,28
Update = 13300, Epoch = 9, LR = 0,000520, AvgCost = 218,5085, Sent = 761, SentPerMin = 1096,92, WordPerSec = 1010,09
Update = 13380, Epoch = 9, LR = 0,000519, AvgCost = 216,6384, Sent = 1000, SentPerMin = 1108,91, WordPerSec = 1017,72
Update = 13400, Epoch = 10, LR = 0,000518, AvgCost = 209,5668, Sent = 64, SentPerMin = 1163,25, WordPerSec = 1014,21
Update = 13500, Epoch = 10, LR = 0,000516, AvgCost = 219,8535, Sent = 352, SentPerMin = 1094,55, WordPerSec = 1006,55
Update = 13600, Epoch = 10, LR = 0,000514, AvgCost = 217,0899, Sent = 645, SentPerMin = 1098,65, WordPerSec = 1011,27
Update = 13700, Epoch = 10, LR = 0,000513, AvgCost = 216,1374, Sent = 948, SentPerMin = 1107,92, WordPerSec = 1018,51
Update = 13718, Epoch = 10, LR = 0,000512, AvgCost = 215,6939, Sent = 1000, SentPerMin = 1109,31, WordPerSec = 1018,09
Update = 13800, Epoch = 11, LR = 0,000511, AvgCost = 217,7100, Sent = 239, SentPerMin = 903,18, WordPerSec = 824,83
Update = 13900, Epoch = 11, LR = 0,000509, AvgCost = 217,2619, Sent = 536, SentPerMin = 900,87, WordPerSec = 828,07
Update = 14000, Epoch = 11, LR = 0,000507, AvgCost = 215,8248, Sent = 831, SentPerMin = 903,43, WordPerSec = 829,56
Update = 14056, Epoch = 11, LR = 0,000506, AvgCost = 215,1251, Sent = 1000, SentPerMin = 907,41, WordPerSec = 832,79
Update = 14100, Epoch = 12, LR = 0,000505, AvgCost = 214,1074, Sent = 131, SentPerMin = 1117,83, WordPerSec = 994,81
Update = 14200, Epoch = 12, LR = 0,000504, AvgCost = 216,6953, Sent = 423, SentPerMin = 1091,46, WordPerSec = 1003,04
Update = 14300, Epoch = 12, LR = 0,000502, AvgCost = 214,0878, Sent = 720, SentPerMin = 1104,24, WordPerSec = 1010,56
Update = 14394, Epoch = 12, LR = 0,000500, AvgCost = 214,3890, Sent = 1000, SentPerMin = 1105,71, WordPerSec = 1014,78
Update = 14400, Epoch = 13, LR = 0,000500, AvgCost = 215,5099, Sent = 16, SentPerMin = 989,85, WordPerSec = 966,14
Update = 14500, Epoch = 13, LR = 0,000498, AvgCost = 221,8767, Sent = 304, SentPerMin = 1080,04, WordPerSec = 1008,75
Update = 14600, Epoch = 13, LR = 0,000497, AvgCost = 214,9671, Sent = 606, SentPerMin = 1097,02, WordPerSec = 1008,23
Update = 14700, Epoch = 13, LR = 0,000495, AvgCost = 214,8153, Sent = 902, SentPerMin = 1101,22, WordPerSec = 1014,21
Update = 14732, Epoch = 13, LR = 0,000494, AvgCost = 213,5827, Sent = 1000, SentPerMin = 1107,78, WordPerSec = 1016,68
Update = 14800, Epoch = 14, LR = 0,000493, AvgCost = 219,1712, Sent = 195, SentPerMin = 1086,58, WordPerSec = 998,72
Update = 14900, Epoch = 14, LR = 0,000492, AvgCost = 214,1724, Sent = 495, SentPerMin = 1103,12, WordPerSec = 1008,15
Update = 15000, Epoch = 14, LR = 0,000490, AvgCost = 214,9182, Sent = 789, SentPerMin = 1098,42, WordPerSec = 1012,82
Update = 15070, Epoch = 14, LR = 0,000489, AvgCost = 212,6338, Sent = 1000, SentPerMin = 1109,42, WordPerSec = 1018,19
Update = 15100, Epoch = 15, LR = 0,000488, AvgCost = 211,0122, Sent = 92, SentPerMin = 1115,84, WordPerSec = 990,31
Update = 15200, Epoch = 15, LR = 0,000487, AvgCost = 215,6455, Sent = 381, SentPerMin = 1093,62, WordPerSec = 1006,36
Update = 15300, Epoch = 15, LR = 0,000485, AvgCost = 213,1842, Sent = 676, SentPerMin = 1103,44, WordPerSec = 1013,04
Update = 15400, Epoch = 15, LR = 0,000483, AvgCost = 211,0562, Sent = 979, SentPerMin = 1112,34, WordPerSec = 1018,53
Update = 15408, Epoch = 15, LR = 0,000483, AvgCost = 212,1046, Sent = 1000, SentPerMin = 1109,84, WordPerSec = 1018,58
Update = 15500, Epoch = 16, LR = 0,000482, AvgCost = 219,2719, Sent = 265, SentPerMin = 1084,11, WordPerSec = 1006,11
Update = 15600, Epoch = 16, LR = 0,000480, AvgCost = 213,2938, Sent = 566, SentPerMin = 1095,21, WordPerSec = 1008,01
Update = 15700, Epoch = 16, LR = 0,000479, AvgCost = 212,7279, Sent = 859, SentPerMin = 1097,29, WordPerSec = 1011,15
Update = 15746, Epoch = 16, LR = 0,000478, AvgCost = 211,3481, Sent = 1000, SentPerMin = 1107,09, WordPerSec = 1016,05
Update = 15800, Epoch = 17, LR = 0,000477, AvgCost = 214,7497, Sent = 157, SentPerMin = 1088,94, WordPerSec = 990,56
Update = 15900, Epoch = 17, LR = 0,000476, AvgCost = 215,3479, Sent = 450, SentPerMin = 1085,49, WordPerSec = 1000,94
Update = 16000, Epoch = 17, LR = 0,000474, AvgCost = 210,9421, Sent = 752, SentPerMin = 1099,61, WordPerSec = 1008,15
Update = 16084, Epoch = 17, LR = 0,000473, AvgCost = 210,7847, Sent = 1000, SentPerMin = 1106,46, WordPerSec = 1015,47
Update = 16100, Epoch = 18, LR = 0,000473, AvgCost = 204,6105, Sent = 51, SentPerMin = 1141,60, WordPerSec = 984,16
Update = 16200, Epoch = 18, LR = 0,000471, AvgCost = 216,8084, Sent = 336, SentPerMin = 1086,78, WordPerSec = 1006,89
Update = 16300, Epoch = 18, LR = 0,000470, AvgCost = 210,8325, Sent = 636, SentPerMin = 1098,86, WordPerSec = 1006,80
Update = 16400, Epoch = 18, LR = 0,000469, AvgCost = 210,8411, Sent = 935, SentPerMin = 1103,86, WordPerSec = 1014,72
Update = 16422, Epoch = 18, LR = 0,000468, AvgCost = 210,3251, Sent = 1000, SentPerMin = 1108,25, WordPerSec = 1017,12
Update = 16500, Epoch = 19, LR = 0,000467, AvgCost = 214,3483, Sent = 225, SentPerMin = 1096,98, WordPerSec = 1008,41
Update = 16600, Epoch = 19, LR = 0,000466, AvgCost = 212,6778, Sent = 524, SentPerMin = 1098,97, WordPerSec = 1011,20
Update = 16700, Epoch = 19, LR = 0,000464, AvgCost = 210,9910, Sent = 819, SentPerMin = 1103,87, WordPerSec = 1015,34
Update = 16760, Epoch = 19, LR = 0,000463, AvgCost = 209,7338, Sent = 1000, SentPerMin = 1112,12, WordPerSec = 1020,67
Update = 16800, Epoch = 20, LR = 0,000463, AvgCost = 206,5591, Sent = 122, SentPerMin = 1141,13, WordPerSec = 996,00
Update = 16900, Epoch = 20, LR = 0,000462, AvgCost = 212,7103, Sent = 411, SentPerMin = 1096,53, WordPerSec = 1009,91
Update = 17000, Epoch = 20, LR = 0,000460, AvgCost = 208,8243, Sent = 709, SentPerMin = 1110,89, WordPerSec = 1014,73
Update = 17098, Epoch = 20, LR = 0,000459, AvgCost = 209,2296, Sent = 1000, SentPerMin = 1110,62, WordPerSec = 1019,29
Update = 17100, Epoch = 21, LR = 0,000459, AvgCost = 185,4395, Sent = 6, SentPerMin = 1097,05, WordPerSec = 996,49
Update = 17200, Epoch = 21, LR = 0,000457, AvgCost = 218,4458, Sent = 292, SentPerMin = 1076,27, WordPerSec = 1006,98
Update = 17300, Epoch = 21, LR = 0,000456, AvgCost = 210,9838, Sent = 595, SentPerMin = 1094,06, WordPerSec = 1006,96
Update = 17400, Epoch = 21, LR = 0,000455, AvgCost = 210,1314, Sent = 890, SentPerMin = 1100,56, WordPerSec = 1014,51
Update = 17436, Epoch = 21, LR = 0,000454, AvgCost = 208,6985, Sent = 1000, SentPerMin = 1107,29, WordPerSec = 1016,24
Update = 17500, Epoch = 22, LR = 0,000454, AvgCost = 215,7819, Sent = 184, SentPerMin = 890,14, WordPerSec = 817,81
Update = 17600, Epoch = 22, LR = 0,000452, AvgCost = 211,4448, Sent = 482, SentPerMin = 898,14, WordPerSec = 825,72
Update = 17700, Epoch = 22, LR = 0,000451, AvgCost = 211,5174, Sent = 775, SentPerMin = 895,65, WordPerSec = 829,83
Update = 17774, Epoch = 22, LR = 0,000450, AvgCost = 208,3881, Sent = 1000, SentPerMin = 908,71, WordPerSec = 833,99
Update = 17800, Epoch = 23, LR = 0,000450, AvgCost = 203,2916, Sent = 81, SentPerMin = 1120,94, WordPerSec = 995,00
Update = 17900, Epoch = 23, LR = 0,000448, AvgCost = 212,5983, Sent = 368, SentPerMin = 1082,49, WordPerSec = 1001,79
Update = 18000, Epoch = 23, LR = 0,000447, AvgCost = 208,7879, Sent = 663, SentPerMin = 1097,41, WordPerSec = 1008,27
Update = 18100, Epoch = 23, LR = 0,000446, AvgCost = 206,5877, Sent = 969, SentPerMin = 1111,17, WordPerSec = 1016,09
Update = 18112, Epoch = 23, LR = 0,000446, AvgCost = 207,4543, Sent = 1000, SentPerMin = 1108,09, WordPerSec = 1016,97
Update = 18200, Epoch = 24, LR = 0,000445, AvgCost = 216,5629, Sent = 251, SentPerMin = 1079,73, WordPerSec = 1008,68
Update = 18300, Epoch = 24, LR = 0,000444, AvgCost = 207,5125, Sent = 556, SentPerMin = 1103,30, WordPerSec = 1009,87
Update = 18400, Epoch = 24, LR = 0,000442, AvgCost = 207,8397, Sent = 847, SentPerMin = 1100,66, WordPerSec = 1014,48
Update = 18450, Epoch = 24, LR = 0,000442, AvgCost = 206,5390, Sent = 1000, SentPerMin = 1109,65, WordPerSec = 1018,40
Update = 18500, Epoch = 25, LR = 0,000441, AvgCost = 208,7596, Sent = 145, SentPerMin = 1086,36, WordPerSec = 988,09
Update = 18600, Epoch = 25, LR = 0,000440, AvgCost = 211,4963, Sent = 437, SentPerMin = 1085,18, WordPerSec = 1005,63
Update = 18700, Epoch = 25, LR = 0,000439, AvgCost = 205,8740, Sent = 739, SentPerMin = 1100,82, WordPerSec = 1009,11
Update = 18788, Epoch = 25, LR = 0,000438, AvgCost = 205,7577, Sent = 1000, SentPerMin = 1107,39, WordPerSec = 1016,33
Update = 18800, Epoch = 26, LR = 0,000438, AvgCost = 167,2354, Sent = 41, SentPerMin = 1249,46, WordPerSec = 961,98
Update = 18900, Epoch = 26, LR = 0,000436, AvgCost = 212,0446, Sent = 324, SentPerMin = 1089,06, WordPerSec = 1011,58
Update = 19000, Epoch = 26, LR = 0,000435, AvgCost = 206,4164, Sent = 624, SentPerMin = 1103,14, WordPerSec = 1011,30
Update = 19100, Epoch = 26, LR = 0,000434, AvgCost = 205,8511, Sent = 921, SentPerMin = 1107,77, WordPerSec = 1019,20
Update = 19126, Epoch = 26, LR = 0,000434, AvgCost = 205,0187, Sent = 1000, SentPerMin = 1111,38, WordPerSec = 1019,99
Update = 19200, Epoch = 27, LR = 0,000433, AvgCost = 210,1883, Sent = 213, SentPerMin = 1089,70, WordPerSec = 1005,29
Update = 19300, Epoch = 27, LR = 0,000432, AvgCost = 208,0144, Sent = 511, SentPerMin = 1093,78, WordPerSec = 1008,59
Update = 19400, Epoch = 27, LR = 0,000431, AvgCost = 206,1415, Sent = 808, SentPerMin = 1103,82, WordPerSec = 1016,75
Update = 19464, Epoch = 27, LR = 0,000430, AvgCost = 204,3558, Sent = 1000, SentPerMin = 1111,88, WordPerSec = 1020,44
Update = 19500, Epoch = 28, LR = 0,000430, AvgCost = 193,3832, Sent = 113, SentPerMin = 1151,05, WordPerSec = 986,03
Update = 19600, Epoch = 28, LR = 0,000429, AvgCost = 207,2917, Sent = 399, SentPerMin = 1093,96, WordPerSec = 1006,95
Update = 19700, Epoch = 28, LR = 0,000427, AvgCost = 204,6014, Sent = 696, SentPerMin = 1110,27, WordPerSec = 1017,24
Update = 19800, Epoch = 28, LR = 0,000426, AvgCost = 204,0048, Sent = 995, SentPerMin = 1112,19, WordPerSec = 1020,03
Update = 19802, Epoch = 28, LR = 0,000426, AvgCost = 203,9471, Sent = 1000, SentPerMin = 1111,40, WordPerSec = 1020,00
Update = 19900, Epoch = 29, LR = 0,000425, AvgCost = 213,6860, Sent = 280, SentPerMin = 1075,97, WordPerSec = 1009,62
Update = 20000, Epoch = 29, LR = 0,000424, AvgCost = 204,5575, Sent = 588, SentPerMin = 1107,33, WordPerSec = 1012,92
Update = 20100, Epoch = 29, LR = 0,000423, AvgCost = 204,8265, Sent = 877, SentPerMin = 785,84, WordPerSec = 724,75
Update = 20140, Epoch = 29, LR = 0,000423, AvgCost = 203,5204, Sent = 1000, SentPerMin = 818,23, WordPerSec = 750,94
Update = 20200, Epoch = 30, LR = 0,000422, AvgCost = 205,7482, Sent = 173, SentPerMin = 1074,79, WordPerSec = 988,85
Update = 20300, Epoch = 30, LR = 0,000421, AvgCost = 207,1832, Sent = 469, SentPerMin = 1089,22, WordPerSec = 1004,72
Update = 20400, Epoch = 30, LR = 0,000420, AvgCost = 205,6151, Sent = 766, SentPerMin = 1094,26, WordPerSec = 1009,71
Update = 20478, Epoch = 30, LR = 0,000419, AvgCost = 203,0889, Sent = 1000, SentPerMin = 1106,02, WordPerSec = 1015,07
Update = 20500, Epoch = 31, LR = 0,000419, AvgCost = 194,2761, Sent = 71, SentPerMin = 1154,22, WordPerSec = 998,97
Update = 20600, Epoch = 31, LR = 0,000418, AvgCost = 206,4905, Sent = 358, SentPerMin = 1085,90, WordPerSec = 997,48
Update = 20700, Epoch = 31, LR = 0,000417, AvgCost = 204,1822, Sent = 652, SentPerMin = 1093,67, WordPerSec = 1004,30
Update = 20800, Epoch = 31, LR = 0,000416, AvgCost = 202,8157, Sent = 955, SentPerMin = 1104,43, WordPerSec = 1013,24
Update = 20816, Epoch = 31, LR = 0,000416, AvgCost = 202,7342, Sent = 1000, SentPerMin = 1104,49, WordPerSec = 1013,67
Update = 20900, Epoch = 32, LR = 0,000415, AvgCost = 204,4981, Sent = 244, SentPerMin = 1101,48, WordPerSec = 1006,30
Update = 21000, Epoch = 32, LR = 0,000414, AvgCost = 204,4341, Sent = 542, SentPerMin = 1093,95, WordPerSec = 1004,74
Update = 21100, Epoch = 32, LR = 0,000413, AvgCost = 203,7027, Sent = 835, SentPerMin = 1097,25, WordPerSec = 1009,49
Update = 21154, Epoch = 32, LR = 0,000413, AvgCost = 202,4502, Sent = 1000, SentPerMin = 1104,76, WordPerSec = 1013,91
Update = 21200, Epoch = 33, LR = 0,000412, AvgCost = 201,0405, Sent = 136, SentPerMin = 907,17, WordPerSec = 807,23
Update = 21300, Epoch = 33, LR = 0,000411, AvgCost = 206,0365, Sent = 428, SentPerMin = 890,50, WordPerSec = 819,06
Update = 21400, Epoch = 33, LR = 0,000410, AvgCost = 203,6335, Sent = 725, SentPerMin = 900,99, WordPerSec = 825,91
Update = 21492, Epoch = 33, LR = 0,000409, AvgCost = 203,1908, Sent = 1000, SentPerMin = 904,01, WordPerSec = 829,67
Update = 21500, Epoch = 34, LR = 0,000409, AvgCost = 183,9653, Sent = 23, SentPerMin = 1065,55, WordPerSec = 928,88
Update = 21600, Epoch = 34, LR = 0,000408, AvgCost = 209,0429, Sent = 311, SentPerMin = 1081,91, WordPerSec = 1005,90
Update = 21700, Epoch = 34, LR = 0,000407, AvgCost = 204,5673, Sent = 611, SentPerMin = 1093,42, WordPerSec = 1007,08
Update = 21800, Epoch = 34, LR = 0,000406, AvgCost = 203,7660, Sent = 906, SentPerMin = 1097,32, WordPerSec = 1012,28
Update = 21830, Epoch = 34, LR = 0,000406, AvgCost = 202,2790, Sent = 1000, SentPerMin = 1104,62, WordPerSec = 1013,78
Update = 21900, Epoch = 35, LR = 0,000405, AvgCost = 206,2812, Sent = 200, SentPerMin = 1074,73, WordPerSec = 986,24
Update = 22000, Epoch = 35, LR = 0,000405, AvgCost = 203,2899, Sent = 501, SentPerMin = 1092,47, WordPerSec = 1000,85
Update = 22100, Epoch = 35, LR = 0,000404, AvgCost = 203,5284, Sent = 795, SentPerMin = 1093,62, WordPerSec = 1007,66
Update = 22168, Epoch = 35, LR = 0,000403, AvgCost = 201,4407, Sent = 1000, SentPerMin = 1104,23, WordPerSec = 1013,43
Update = 22200, Epoch = 36, LR = 0,000403, AvgCost = 195,0031, Sent = 99, SentPerMin = 1130,83, WordPerSec = 985,38
Update = 22300, Epoch = 36, LR = 0,000402, AvgCost = 203,1631, Sent = 388, SentPerMin = 1092,83, WordPerSec = 1001,62
Update = 22400, Epoch = 36, LR = 0,000401, AvgCost = 201,3569, Sent = 683, SentPerMin = 1089,96, WordPerSec = 998,73
Update = 22500, Epoch = 36, LR = 0,000400, AvgCost = 200,8302, Sent = 984, SentPerMin = 1061,04, WordPerSec = 973,72
Update = 22506, Epoch = 36, LR = 0,000400, AvgCost = 200,6748, Sent = 1000, SentPerMin = 1061,36, WordPerSec = 974,08
Update = 22600, Epoch = 37, LR = 0,000399, AvgCost = 208,2829, Sent = 269, SentPerMin = 1069,05, WordPerSec = 999,24
Update = 22700, Epoch = 37, LR = 0,000398, AvgCost = 201,7675, Sent = 573, SentPerMin = 1092,77, WordPerSec = 1003,62
Update = 22800, Epoch = 37, LR = 0,000397, AvgCost = 201,0870, Sent = 865, SentPerMin = 1094,25, WordPerSec = 1007,85
Update = 22844, Epoch = 37, LR = 0,000397, AvgCost = 200,0432, Sent = 1000, SentPerMin = 1102,07, WordPerSec = 1011,45
Update = 22900, Epoch = 38, LR = 0,000396, AvgCost = 203,5402, Sent = 161, SentPerMin = 1071,30, WordPerSec = 982,69
Update = 23000, Epoch = 38, LR = 0,000396, AvgCost = 204,4335, Sent = 456, SentPerMin = 1080,96, WordPerSec = 998,78
Update = 23100, Epoch = 38, LR = 0,000395, AvgCost = 200,9096, Sent = 757, SentPerMin = 1090,68, WordPerSec = 1003,23
Update = 23182, Epoch = 38, LR = 0,000394, AvgCost = 199,6082, Sent = 1000, SentPerMin = 1102,10, WordPerSec = 1011,47
Update = 23200, Epoch = 39, LR = 0,000394, AvgCost = 185,6967, Sent = 59, SentPerMin = 1176,59, WordPerSec = 983,15
Update = 23300, Epoch = 39, LR = 0,000393, AvgCost = 204,1168, Sent = 343, SentPerMin = 1081,38, WordPerSec = 999,83
Update = 23400, Epoch = 39, LR = 0,000392, AvgCost = 200,4294, Sent = 640, SentPerMin = 1090,76, WordPerSec = 1002,33
Update = 23500, Epoch = 39, LR = 0,000391, AvgCost = 199,2109, Sent = 943, SentPerMin = 1101,59, WordPerSec = 1011,13
Update = 23520, Epoch = 39, LR = 0,000391, AvgCost = 199,0986, Sent = 1000, SentPerMin = 1103,34, WordPerSec = 1012,61
Update = 23600, Epoch = 40, LR = 0,000391, AvgCost = 201,9228, Sent = 231, SentPerMin = 1082,97, WordPerSec = 996,56
Update = 23700, Epoch = 40, LR = 0,000390, AvgCost = 201,3998, Sent = 529, SentPerMin = 1082,68, WordPerSec = 996,48
Update = 23800, Epoch = 40, LR = 0,000389, AvgCost = 199,9232, Sent = 824, SentPerMin = 1087,68, WordPerSec = 1000,86
Update = 23858, Epoch = 40, LR = 0,000388, AvgCost = 198,6517, Sent = 1000, SentPerMin = 1096,50, WordPerSec = 1006,33
Update = 23900, Epoch = 41, LR = 0,000388, AvgCost = 195,6268, Sent = 126, SentPerMin = 1114,67, WordPerSec = 984,03
Update = 24000, Epoch = 41, LR = 0,000387, AvgCost = 200,9153, Sent = 418, SentPerMin = 1091,01, WordPerSec = 1001,14
Update = 24100, Epoch = 41, LR = 0,000386, AvgCost = 198,2534, Sent = 715, SentPerMin = 1105,70, WordPerSec = 1011,16
Update = 24196, Epoch = 41, LR = 0,000386, AvgCost = 198,3585, Sent = 1000, SentPerMin = 1105,28, WordPerSec = 1014,39
Update = 24200, Epoch = 42, LR = 0,000386, AvgCost = 175,1654, Sent = 12, SentPerMin = 1103,86, WordPerSec = 993,48
Update = 24300, Epoch = 42, LR = 0,000385, AvgCost = 205,7153, Sent = 299, SentPerMin = 1078,80, WordPerSec = 1006,28
Update = 24400, Epoch = 42, LR = 0,000384, AvgCost = 199,6810, Sent = 601, SentPerMin = 1094,94, WordPerSec = 1005,85
Update = 24500, Epoch = 42, LR = 0,000383, AvgCost = 198,7033, Sent = 897, SentPerMin = 1100,38, WordPerSec = 1012,16
Update = 24534, Epoch = 42, LR = 0,000383, AvgCost = 197,8153, Sent = 1000, SentPerMin = 1105,03, WordPerSec = 1014,16
Update = 24600, Epoch = 43, LR = 0,000383, AvgCost = 203,9343, Sent = 189, SentPerMin = 1074,22, WordPerSec = 989,53
Update = 24700, Epoch = 43, LR = 0,000382, AvgCost = 199,5632, Sent = 490, SentPerMin = 1094,42, WordPerSec = 1000,50
Update = 24800, Epoch = 43, LR = 0,000381, AvgCost = 199,6818, Sent = 784, SentPerMin = 1090,79, WordPerSec = 1005,30
Update = 24872, Epoch = 43, LR = 0,000380, AvgCost = 197,4502, Sent = 1000, SentPerMin = 1101,85, WordPerSec = 1011,24
Update = 24900, Epoch = 44, LR = 0,000380, AvgCost = 191,6085, Sent = 87, SentPerMin = 923,60, WordPerSec = 811,60
Update = 25000, Epoch = 44, LR = 0,000379, AvgCost = 202,1769, Sent = 375, SentPerMin = 889,07, WordPerSec = 819,45
Update = 25100, Epoch = 44, LR = 0,000379, AvgCost = 200,2375, Sent = 668, SentPerMin = 892,84, WordPerSec = 822,73
Update = 25200, Epoch = 44, LR = 0,000378, AvgCost = 197,6233, Sent = 974, SentPerMin = 905,37, WordPerSec = 828,48
Update = 25210, Epoch = 44, LR = 0,000378, AvgCost = 198,5850, Sent = 1000, SentPerMin = 903,10, WordPerSec = 828,84
Update = 25300, Epoch = 45, LR = 0,000377, AvgCost = 205,5825, Sent = 258, SentPerMin = 1074,47, WordPerSec = 1001,52
Update = 25400, Epoch = 45, LR = 0,000376, AvgCost = 198,6507, Sent = 562, SentPerMin = 1097,93, WordPerSec = 1006,73
Update = 25500, Epoch = 45, LR = 0,000376, AvgCost = 198,7613, Sent = 854, SentPerMin = 1095,44, WordPerSec = 1008,77
Update = 25548, Epoch = 45, LR = 0,000375, AvgCost = 197,6852, Sent = 1000, SentPerMin = 1104,04, WordPerSec = 1013,25
Update = 25600, Epoch = 46, LR = 0,000375, AvgCost = 196,5035, Sent = 152, SentPerMin = 1096,84, WordPerSec = 986,80
Update = 25700, Epoch = 46, LR = 0,000374, AvgCost = 201,9460, Sent = 443, SentPerMin = 1075,96, WordPerSec = 994,80
Update = 25800, Epoch = 46, LR = 0,000374, AvgCost = 197,6138, Sent = 744, SentPerMin = 1090,30, WordPerSec = 1001,37
Update = 25886, Epoch = 46, LR = 0,000373, AvgCost = 197,1667, Sent = 1000, SentPerMin = 1099,95, WordPerSec = 1009,50
Update = 25900, Epoch = 47, LR = 0,000373, AvgCost = 182,1841, Sent = 45, SentPerMin = 1143,72, WordPerSec = 936,58
Update = 26000, Epoch = 47, LR = 0,000372, AvgCost = 201,9421, Sent = 330, SentPerMin = 1077,22, WordPerSec = 998,55
Update = 26100, Epoch = 47, LR = 0,000371, AvgCost = 197,7571, Sent = 630, SentPerMin = 1093,07, WordPerSec = 1002,42
Update = 26200, Epoch = 47, LR = 0,000371, AvgCost = 196,9899, Sent = 929, SentPerMin = 1100,37, WordPerSec = 1010,96
Update = 26224, Epoch = 47, LR = 0,000371, AvgCost = 196,5329, Sent = 1000, SentPerMin = 1103,70, WordPerSec = 1012,94
Update = 26300, Epoch = 48, LR = 0,000370, AvgCost = 200,1125, Sent = 219, SentPerMin = 1079,23, WordPerSec = 994,46
Update = 26400, Epoch = 48, LR = 0,000369, AvgCost = 198,4409, Sent = 519, SentPerMin = 1091,89, WordPerSec = 1001,95
Update = 26500, Epoch = 48, LR = 0,000369, AvgCost = 197,4574, Sent = 814, SentPerMin = 1094,23, WordPerSec = 1006,47
Update = 26562, Epoch = 48, LR = 0,000368, AvgCost = 196,1550, Sent = 1000, SentPerMin = 1102,39, WordPerSec = 1011,73
Update = 26600, Epoch = 49, LR = 0,000368, AvgCost = 181,4189, Sent = 119, SentPerMin = 1147,89, WordPerSec = 976,51
Update = 26700, Epoch = 49, LR = 0,000367, AvgCost = 198,1345, Sent = 406, SentPerMin = 1088,14, WordPerSec = 1000,10
Update = 26800, Epoch = 49, LR = 0,000367, AvgCost = 195,7474, Sent = 702, SentPerMin = 1101,38, WordPerSec = 1007,25
Update = 26900, Epoch = 49, LR = 0,000366, AvgCost = 195,7333, Sent = 1000, SentPerMin = 1103,71, WordPerSec = 1012,94
Update = 26900, Epoch = 49, LR = 0,000366, AvgCost = 195,7333, Sent = 1000, SentPerMin = 1103,69, WordPerSec = 1012,93
Update = 27000, Epoch = 50, LR = 0,000365, AvgCost = 204,5642, Sent = 285, SentPerMin = 1067,18, WordPerSec = 1002,09
Update = 27100, Epoch = 50, LR = 0,000364, AvgCost = 196,4376, Sent = 592, SentPerMin = 1097,01, WordPerSec = 1006,30
Update = 27200, Epoch = 50, LR = 0,000364, AvgCost = 197,0469, Sent = 882, SentPerMin = 1093,03, WordPerSec = 1009,77
Update = 27238, Epoch = 50, LR = 0,000364, AvgCost = 195,3140, Sent = 1000, SentPerMin = 1103,22, WordPerSec = 1012,50
Update = 27300, Epoch = 51, LR = 0,000363, AvgCost = 197,2282, Sent = 178, SentPerMin = 1059,16, WordPerSec = 977,34
Update = 27400, Epoch = 51, LR = 0,000362, AvgCost = 198,3918, Sent = 475, SentPerMin = 1075,90, WordPerSec = 991,03
Update = 27500, Epoch = 51, LR = 0,000362, AvgCost = 197,2989, Sent = 771, SentPerMin = 1083,58, WordPerSec = 1001,55
Update = 27576, Epoch = 51, LR = 0,000361, AvgCost = 194,7813, Sent = 1000, SentPerMin = 1098,85, WordPerSec = 1008,49
Update = 27600, Epoch = 52, LR = 0,000361, AvgCost = 188,2986, Sent = 76, SentPerMin = 1128,09, WordPerSec = 995,00
Update = 27700, Epoch = 52, LR = 0,000361, AvgCost = 198,5853, Sent = 363, SentPerMin = 1089,12, WordPerSec = 1004,31
Update = 27800, Epoch = 52, LR = 0,000360, AvgCost = 196,1671, Sent = 656, SentPerMin = 1090,43, WordPerSec = 1004,00
Update = 27900, Epoch = 52, LR = 0,000359, AvgCost = 194,0926, Sent = 961, SentPerMin = 1100,70, WordPerSec = 1008,86
Update = 27914, Epoch = 52, LR = 0,000359, AvgCost = 194,4049, Sent = 1000, SentPerMin = 1100,42, WordPerSec = 1009,93
Update = 28000, Epoch = 53, LR = 0,000359, AvgCost = 197,9756, Sent = 248, SentPerMin = 1078,03, WordPerSec = 995,88
Update = 28100, Epoch = 53, LR = 0,000358, AvgCost = 195,0372, Sent = 549, SentPerMin = 1094,59, WordPerSec = 1002,91
Update = 28200, Epoch = 53, LR = 0,000357, AvgCost = 195,0732, Sent = 841, SentPerMin = 1093,14, WordPerSec = 1006,62
Update = 28252, Epoch = 53, LR = 0,000357, AvgCost = 193,8509, Sent = 1000, SentPerMin = 1102,22, WordPerSec = 1011,58
Update = 28300, Epoch = 54, LR = 0,000357, AvgCost = 191,7741, Sent = 140, SentPerMin = 1097,33, WordPerSec = 984,98
Update = 28400, Epoch = 54, LR = 0,000356, AvgCost = 199,3309, Sent = 431, SentPerMin = 1083,69, WordPerSec = 1004,52
Update = 28500, Epoch = 54, LR = 0,000355, AvgCost = 193,3410, Sent = 732, SentPerMin = 1101,58, WordPerSec = 1007,13
Update = 28590, Epoch = 54, LR = 0,000355, AvgCost = 193,5833, Sent = 1000, SentPerMin = 1102,85, WordPerSec = 1012,16
Update = 28600, Epoch = 55, LR = 0,000355, AvgCost = 160,8969, Sent = 33, SentPerMin = 990,32, WordPerSec = 765,25
Update = 28700, Epoch = 55, LR = 0,000354, AvgCost = 201,4184, Sent = 317, SentPerMin = 881,75, WordPerSec = 821,11
Update = 28800, Epoch = 55, LR = 0,000354, AvgCost = 196,3257, Sent = 618, SentPerMin = 896,53, WordPerSec = 822,81
Update = 28900, Epoch = 55, LR = 0,000353, AvgCost = 196,0040, Sent = 914, SentPerMin = 897,58, WordPerSec = 827,12
Update = 28928, Epoch = 55, LR = 0,000353, AvgCost = 194,9566, Sent = 1000, SentPerMin = 901,68, WordPerSec = 827,53
Update = 29000, Epoch = 56, LR = 0,000352, AvgCost = 201,4809, Sent = 205, SentPerMin = 1061,04, WordPerSec = 990,65
Update = 29100, Epoch = 56, LR = 0,000352, AvgCost = 195,8270, Sent = 506, SentPerMin = 1090,82, WordPerSec = 1000,53
Update = 29200, Epoch = 56, LR = 0,000351, AvgCost = 195,7310, Sent = 802, SentPerMin = 1093,55, WordPerSec = 1006,76
Update = 29266, Epoch = 56, LR = 0,000351, AvgCost = 194,0935, Sent = 1000, SentPerMin = 1103,01, WordPerSec = 1012,31
Update = 29300, Epoch = 57, LR = 0,000351, AvgCost = 184,5382, Sent = 106, SentPerMin = 1140,65, WordPerSec = 982,46
Update = 29400, Epoch = 57, LR = 0,000350, AvgCost = 196,8979, Sent = 393, SentPerMin = 1086,76, WordPerSec = 1001,59
Update = 29500, Epoch = 57, LR = 0,000349, AvgCost = 194,0628, Sent = 689, SentPerMin = 1104,10, WordPerSec = 1012,10
Update = 29600, Epoch = 57, LR = 0,000349, AvgCost = 193,7743, Sent = 989, SentPerMin = 1106,84, WordPerSec = 1016,10
Update = 29604, Epoch = 57, LR = 0,000349, AvgCost = 193,6209, Sent = 1000, SentPerMin = 1106,89, WordPerSec = 1015,87
Update = 29700, Epoch = 58, LR = 0,000348, AvgCost = 203,0984, Sent = 275, SentPerMin = 1073,65, WordPerSec = 1007,86
Update = 29800, Epoch = 58, LR = 0,000348, AvgCost = 194,4171, Sent = 581, SentPerMin = 1099,08, WordPerSec = 1008,24
Update = 29900, Epoch = 58, LR = 0,000347, AvgCost = 194,5514, Sent = 870, SentPerMin = 1096,49, WordPerSec = 1012,36
Update = 29942, Epoch = 58, LR = 0,000347, AvgCost = 193,1998, Sent = 1000, SentPerMin = 1105,61, WordPerSec = 1014,69
Update = 30000, Epoch = 59, LR = 0,000346, AvgCost = 196,2056, Sent = 166, SentPerMin = 1062,21, WordPerSec = 978,81
Update = 30100, Epoch = 59, LR = 0,000346, AvgCost = 196,7713, Sent = 463, SentPerMin = 817,95, WordPerSec = 754,97
Update = 30200, Epoch = 59, LR = 0,000345, AvgCost = 194,6211, Sent = 761, SentPerMin = 910,02, WordPerSec = 837,99
Update = 30280, Epoch = 59, LR = 0,000345, AvgCost = 192,7652, Sent = 1000, SentPerMin = 955,16, WordPerSec = 876,62
Update = 30300, Epoch = 60, LR = 0,000345, AvgCost = 187,4178, Sent = 64, SentPerMin = 1149,93, WordPerSec = 1002,60
Update = 30400, Epoch = 60, LR = 0,000344, AvgCost = 196,0960, Sent = 352, SentPerMin = 1086,34, WordPerSec = 999,00
Update = 30500, Epoch = 60, LR = 0,000344, AvgCost = 193,9087, Sent = 645, SentPerMin = 1093,93, WordPerSec = 1006,92
Update = 30600, Epoch = 60, LR = 0,000343, AvgCost = 192,7282, Sent = 948, SentPerMin = 1096,05, WordPerSec = 1007,60
Update = 30618, Epoch = 60, LR = 0,000343, AvgCost = 192,3105, Sent = 1000, SentPerMin = 1097,41, WordPerSec = 1007,17
Update = 30700, Epoch = 61, LR = 0,000342, AvgCost = 194,1841, Sent = 239, SentPerMin = 1103,09, WordPerSec = 1007,40
Update = 30800, Epoch = 61, LR = 0,000342, AvgCost = 194,1254, Sent = 536, SentPerMin = 1101,05, WordPerSec = 1012,07
Update = 30900, Epoch = 61, LR = 0,000341, AvgCost = 192,6506, Sent = 831, SentPerMin = 1104,08, WordPerSec = 1013,80
Update = 30956, Epoch = 61, LR = 0,000341, AvgCost = 191,9683, Sent = 1000, SentPerMin = 1109,77, WordPerSec = 1018,51
Update = 31000, Epoch = 62, LR = 0,000341, AvgCost = 189,8554, Sent = 131, SentPerMin = 1117,43, WordPerSec = 994,46
Update = 31100, Epoch = 62, LR = 0,000340, AvgCost = 193,9672, Sent = 423, SentPerMin = 1096,07, WordPerSec = 1007,28
Update = 31200, Epoch = 62, LR = 0,000340, AvgCost = 191,5996, Sent = 720, SentPerMin = 1111,65, WordPerSec = 1017,34
Update = 31294, Epoch = 62, LR = 0,000339, AvgCost = 191,6955, Sent = 1000, SentPerMin = 1112,62, WordPerSec = 1021,13
Update = 31300, Epoch = 63, LR = 0,000339, AvgCost = 190,1551, Sent = 16, SentPerMin = 1004,93, WordPerSec = 980,85
Update = 31400, Epoch = 63, LR = 0,000339, AvgCost = 198,8227, Sent = 304, SentPerMin = 1081,47, WordPerSec = 1010,08
Update = 31500, Epoch = 63, LR = 0,000338, AvgCost = 193,0121, Sent = 606, SentPerMin = 1101,29, WordPerSec = 1012,15
Update = 31600, Epoch = 63, LR = 0,000338, AvgCost = 192,5368, Sent = 902, SentPerMin = 1105,55, WordPerSec = 1018,20
Update = 31632, Epoch = 63, LR = 0,000337, AvgCost = 191,4101, Sent = 1000, SentPerMin = 1112,36, WordPerSec = 1020,89
Update = 31700, Epoch = 64, LR = 0,000337, AvgCost = 196,1504, Sent = 195, SentPerMin = 1082,65, WordPerSec = 995,11
Update = 31800, Epoch = 64, LR = 0,000336, AvgCost = 192,5289, Sent = 495, SentPerMin = 1103,61, WordPerSec = 1008,59
Update = 31900, Epoch = 64, LR = 0,000336, AvgCost = 193,2046, Sent = 789, SentPerMin = 1099,30, WordPerSec = 1013,63
Update = 31970, Epoch = 64, LR = 0,000336, AvgCost = 191,0445, Sent = 1000, SentPerMin = 1111,25, WordPerSec = 1019,87
Update = 32000, Epoch = 65, LR = 0,000335, AvgCost = 187,6882, Sent = 92, SentPerMin = 1115,61, WordPerSec = 990,11
Update = 32100, Epoch = 65, LR = 0,000335, AvgCost = 193,8480, Sent = 381, SentPerMin = 1097,67, WordPerSec = 1010,08
Update = 32200, Epoch = 65, LR = 0,000334, AvgCost = 192,0035, Sent = 676, SentPerMin = 1104,60, WordPerSec = 1014,10
Update = 32300, Epoch = 65, LR = 0,000334, AvgCost = 189,8676, Sent = 979, SentPerMin = 1113,75, WordPerSec = 1019,82
Update = 32308, Epoch = 65, LR = 0,000334, AvgCost = 190,8070, Sent = 1000, SentPerMin = 1111,29, WordPerSec = 1019,91
Starting inference...
Inference results:
和 。

和和 。

和了的

             */

            /*
             Update = 10100, Epoch = 0, LR = 0,000597, AvgCost = 266,4952, Sent = 285, SentPerMin = 845,40, WordPerSec = 793,84
Update = 10200, Epoch = 0, LR = 0,000594, AvgCost = 247,7676, Sent = 592, SentPerMin = 903,09, WordPerSec = 828,42
Update = 10300, Epoch = 0, LR = 0,000591, AvgCost = 244,6666, Sent = 882, SentPerMin = 911,06, WordPerSec = 841,66
Update = 10338, Epoch = 0, LR = 0,000590, AvgCost = 241,5435, Sent = 1000, SentPerMin = 921,46, WordPerSec = 845,69
Update = 10400, Epoch = 1, LR = 0,000588, AvgCost = 233,5503, Sent = 178, SentPerMin = 1124,44, WordPerSec = 1037,58
Update = 10500, Epoch = 1, LR = 0,000586, AvgCost = 233,8012, Sent = 475, SentPerMin = 1136,60, WordPerSec = 1046,95
Update = 10600, Epoch = 1, LR = 0,000583, AvgCost = 232,1020, Sent = 771, SentPerMin = 1139,01, WordPerSec = 1052,79
Update = 10676, Epoch = 1, LR = 0,000581, AvgCost = 228,9214, Sent = 1000, SentPerMin = 1151,60, WordPerSec = 1056,90
Update = 10700, Epoch = 2, LR = 0,000580, AvgCost = 220,4390, Sent = 76, SentPerMin = 1164,14, WordPerSec = 1026,79
Update = 10800, Epoch = 2, LR = 0,000577, AvgCost = 230,6545, Sent = 363, SentPerMin = 1129,66, WordPerSec = 1041,69
Update = 10900, Epoch = 2, LR = 0,000575, AvgCost = 227,3312, Sent = 656, SentPerMin = 1137,65, WordPerSec = 1047,47
Update = 11000, Epoch = 2, LR = 0,000572, AvgCost = 224,7502, Sent = 961, SentPerMin = 1149,36, WordPerSec = 1053,46
Update = 11014, Epoch = 2, LR = 0,000572, AvgCost = 225,1550, Sent = 1000, SentPerMin = 1149,34, WordPerSec = 1054,83
Update = 11100, Epoch = 3, LR = 0,000569, AvgCost = 228,0804, Sent = 248, SentPerMin = 1123,38, WordPerSec = 1037,76
Update = 11200, Epoch = 3, LR = 0,000567, AvgCost = 224,0980, Sent = 549, SentPerMin = 1138,07, WordPerSec = 1042,74
Update = 11300, Epoch = 3, LR = 0,000564, AvgCost = 223,8847, Sent = 841, SentPerMin = 1138,11, WordPerSec = 1048,03
Update = 11352, Epoch = 3, LR = 0,000563, AvgCost = 222,4978, Sent = 1000, SentPerMin = 1146,54, WordPerSec = 1052,26
Update = 11400, Epoch = 4, LR = 0,000562, AvgCost = 220,2435, Sent = 140, SentPerMin = 1125,46, WordPerSec = 1010,23
Update = 11500, Epoch = 4, LR = 0,000560, AvgCost = 227,3755, Sent = 431, SentPerMin = 1114,71, WordPerSec = 1033,28
Update = 11600, Epoch = 4, LR = 0,000557, AvgCost = 220,4369, Sent = 732, SentPerMin = 1138,38, WordPerSec = 1040,77
Update = 11690, Epoch = 4, LR = 0,000555, AvgCost = 220,7399, Sent = 1000, SentPerMin = 1141,55, WordPerSec = 1047,68
Update = 11700, Epoch = 5, LR = 0,000555, AvgCost = 182,6375, Sent = 33, SentPerMin = 1235,95, WordPerSec = 955,05
Update = 11800, Epoch = 5, LR = 0,000552, AvgCost = 227,1222, Sent = 317, SentPerMin = 1114,25, WordPerSec = 1037,62
Update = 11900, Epoch = 5, LR = 0,000550, AvgCost = 220,8617, Sent = 618, SentPerMin = 1131,10, WordPerSec = 1038,10
Update = 12000, Epoch = 5, LR = 0,000548, AvgCost = 220,4857, Sent = 914, SentPerMin = 1136,19, WordPerSec = 1046,99
Update = 12028, Epoch = 5, LR = 0,000547, AvgCost = 219,3588, Sent = 1000, SentPerMin = 1142,54, WordPerSec = 1048,58
Update = 12100, Epoch = 6, LR = 0,000545, AvgCost = 227,3479, Sent = 205, SentPerMin = 1101,00, WordPerSec = 1027,96
Update = 12200, Epoch = 6, LR = 0,000543, AvgCost = 220,0642, Sent = 506, SentPerMin = 1127,07, WordPerSec = 1033,78
Update = 12300, Epoch = 6, LR = 0,000541, AvgCost = 220,0675, Sent = 802, SentPerMin = 1134,49, WordPerSec = 1044,46
Update = 12366, Epoch = 6, LR = 0,000540, AvgCost = 218,1980, Sent = 1000, SentPerMin = 1143,61, WordPerSec = 1049,57
Update = 12400, Epoch = 7, LR = 0,000539, AvgCost = 208,4906, Sent = 106, SentPerMin = 1174,89, WordPerSec = 1011,95
Update = 12500, Epoch = 7, LR = 0,000537, AvgCost = 220,8289, Sent = 393, SentPerMin = 1121,05, WordPerSec = 1033,19
Update = 12600, Epoch = 7, LR = 0,000535, AvgCost = 217,4594, Sent = 689, SentPerMin = 1141,05, WordPerSec = 1045,96
Update = 12700, Epoch = 7, LR = 0,000532, AvgCost = 217,1124, Sent = 989, SentPerMin = 1143,02, WordPerSec = 1049,31
Update = 12704, Epoch = 7, LR = 0,000532, AvgCost = 216,9628, Sent = 1000, SentPerMin = 1143,13, WordPerSec = 1049,12
Update = 12800, Epoch = 8, LR = 0,000530, AvgCost = 227,1892, Sent = 275, SentPerMin = 1105,85, WordPerSec = 1038,09
Update = 12900, Epoch = 8, LR = 0,000528, AvgCost = 217,2145, Sent = 581, SentPerMin = 1134,47, WordPerSec = 1040,71
Update = 13000, Epoch = 8, LR = 0,000526, AvgCost = 217,7295, Sent = 870, SentPerMin = 1133,99, WordPerSec = 1046,98
Update = 13042, Epoch = 8, LR = 0,000525, AvgCost = 216,2731, Sent = 1000, SentPerMin = 1143,73, WordPerSec = 1049,68
Update = 13100, Epoch = 9, LR = 0,000524, AvgCost = 218,8056, Sent = 166, SentPerMin = 1097,42, WordPerSec = 1011,26
Update = 13200, Epoch = 9, LR = 0,000522, AvgCost = 219,0726, Sent = 463, SentPerMin = 1120,01, WordPerSec = 1033,77
Update = 13300, Epoch = 9, LR = 0,000520, AvgCost = 217,2055, Sent = 761, SentPerMin = 1128,78, WordPerSec = 1039,44
Update = 13380, Epoch = 9, LR = 0,000519, AvgCost = 215,2961, Sent = 1000, SentPerMin = 1140,13, WordPerSec = 1046,37
Update = 13400, Epoch = 10, LR = 0,000518, AvgCost = 209,0206, Sent = 64, SentPerMin = 1176,41, WordPerSec = 1025,69
Update = 13500, Epoch = 10, LR = 0,000516, AvgCost = 218,0907, Sent = 352, SentPerMin = 1124,46, WordPerSec = 1034,05
Update = 13600, Epoch = 10, LR = 0,000514, AvgCost = 215,7262, Sent = 645, SentPerMin = 1129,72, WordPerSec = 1039,87
Update = 13700, Epoch = 10, LR = 0,000513, AvgCost = 214,8579, Sent = 948, SentPerMin = 1139,36, WordPerSec = 1047,42
Update = 13718, Epoch = 10, LR = 0,000512, AvgCost = 214,3743, Sent = 1000, SentPerMin = 1141,78, WordPerSec = 1047,88
Update = 13800, Epoch = 11, LR = 0,000511, AvgCost = 215,1732, Sent = 239, SentPerMin = 1137,89, WordPerSec = 1039,18
Update = 13900, Epoch = 11, LR = 0,000509, AvgCost = 215,5762, Sent = 536, SentPerMin = 1130,07, WordPerSec = 1038,74
Update = 14000, Epoch = 11, LR = 0,000507, AvgCost = 214,4447, Sent = 831, SentPerMin = 1136,37, WordPerSec = 1043,45
Update = 14056, Epoch = 11, LR = 0,000506, AvgCost = 213,6864, Sent = 1000, SentPerMin = 1141,49, WordPerSec = 1047,62
Update = 14100, Epoch = 12, LR = 0,000505, AvgCost = 212,7295, Sent = 131, SentPerMin = 941,82, WordPerSec = 838,18
Update = 14200, Epoch = 12, LR = 0,000504, AvgCost = 215,7882, Sent = 423, SentPerMin = 922,60, WordPerSec = 847,86
Update = 14300, Epoch = 12, LR = 0,000502, AvgCost = 213,0969, Sent = 720, SentPerMin = 936,34, WordPerSec = 856,90
Update = 14394, Epoch = 12, LR = 0,000500, AvgCost = 213,1929, Sent = 1000, SentPerMin = 937,65, WordPerSec = 860,54
Update = 14400, Epoch = 13, LR = 0,000500, AvgCost = 216,5836, Sent = 16, SentPerMin = 1034,65, WordPerSec = 1009,86
Update = 14500, Epoch = 13, LR = 0,000498, AvgCost = 221,2488, Sent = 304, SentPerMin = 1116,80, WordPerSec = 1043,08
Update = 14600, Epoch = 13, LR = 0,000497, AvgCost = 214,2443, Sent = 606, SentPerMin = 1133,48, WordPerSec = 1041,74
Update = 14700, Epoch = 13, LR = 0,000495, AvgCost = 213,7355, Sent = 902, SentPerMin = 1138,45, WordPerSec = 1048,50
Update = 14732, Epoch = 13, LR = 0,000494, AvgCost = 212,4500, Sent = 1000, SentPerMin = 1144,45, WordPerSec = 1050,34
Update = 14800, Epoch = 14, LR = 0,000493, AvgCost = 217,8992, Sent = 195, SentPerMin = 1117,44, WordPerSec = 1027,09
Update = 14900, Epoch = 14, LR = 0,000492, AvgCost = 213,3485, Sent = 495, SentPerMin = 1135,32, WordPerSec = 1037,58
Update = 15000, Epoch = 14, LR = 0,000490, AvgCost = 214,1778, Sent = 789, SentPerMin = 1132,20, WordPerSec = 1043,97
Update = 15070, Epoch = 14, LR = 0,000489, AvgCost = 211,7718, Sent = 1000, SentPerMin = 1143,86, WordPerSec = 1049,80
Update = 15100, Epoch = 15, LR = 0,000488, AvgCost = 210,3463, Sent = 92, SentPerMin = 1136,17, WordPerSec = 1008,35
Update = 15200, Epoch = 15, LR = 0,000487, AvgCost = 215,0027, Sent = 381, SentPerMin = 1121,73, WordPerSec = 1032,23
Update = 15300, Epoch = 15, LR = 0,000485, AvgCost = 212,5174, Sent = 676, SentPerMin = 1134,33, WordPerSec = 1041,40
Update = 15400, Epoch = 15, LR = 0,000483, AvgCost = 210,1770, Sent = 979, SentPerMin = 1145,64, WordPerSec = 1049,02
Update = 15408, Epoch = 15, LR = 0,000483, AvgCost = 211,2341, Sent = 1000, SentPerMin = 1143,18, WordPerSec = 1049,17
Update = 15500, Epoch = 16, LR = 0,000482, AvgCost = 218,1493, Sent = 265, SentPerMin = 1121,31, WordPerSec = 1040,64
Update = 15600, Epoch = 16, LR = 0,000480, AvgCost = 212,8386, Sent = 566, SentPerMin = 1135,35, WordPerSec = 1044,95
Update = 15700, Epoch = 16, LR = 0,000479, AvgCost = 212,0905, Sent = 859, SentPerMin = 1134,25, WordPerSec = 1045,21
Update = 15746, Epoch = 16, LR = 0,000478, AvgCost = 210,7094, Sent = 1000, SentPerMin = 1142,64, WordPerSec = 1048,68
Update = 15800, Epoch = 17, LR = 0,000477, AvgCost = 213,1184, Sent = 157, SentPerMin = 1116,06, WordPerSec = 1015,23
Update = 15900, Epoch = 17, LR = 0,000476, AvgCost = 214,6933, Sent = 450, SentPerMin = 1124,89, WordPerSec = 1037,27
Update = 16000, Epoch = 17, LR = 0,000474, AvgCost = 210,1973, Sent = 752, SentPerMin = 1136,58, WordPerSec = 1042,04
Update = 16084, Epoch = 17, LR = 0,000473, AvgCost = 210,0300, Sent = 1000, SentPerMin = 1142,94, WordPerSec = 1048,95
Update = 16100, Epoch = 18, LR = 0,000473, AvgCost = 205,1890, Sent = 51, SentPerMin = 1192,40, WordPerSec = 1027,96
Update = 16200, Epoch = 18, LR = 0,000471, AvgCost = 215,7026, Sent = 336, SentPerMin = 1122,98, WordPerSec = 1040,43
Update = 16300, Epoch = 18, LR = 0,000470, AvgCost = 210,2952, Sent = 636, SentPerMin = 1136,17, WordPerSec = 1040,98
Update = 16400, Epoch = 18, LR = 0,000469, AvgCost = 209,8806, Sent = 935, SentPerMin = 1140,69, WordPerSec = 1048,58
Update = 16422, Epoch = 18, LR = 0,000468, AvgCost = 209,4404, Sent = 1000, SentPerMin = 1144,60, WordPerSec = 1050,48
Update = 16500, Epoch = 19, LR = 0,000467, AvgCost = 212,5322, Sent = 225, SentPerMin = 1125,32, WordPerSec = 1034,46
Update = 16600, Epoch = 19, LR = 0,000466, AvgCost = 211,6010, Sent = 524, SentPerMin = 1130,95, WordPerSec = 1040,62
Update = 16700, Epoch = 19, LR = 0,000464, AvgCost = 209,8026, Sent = 819, SentPerMin = 1135,83, WordPerSec = 1044,74
Update = 16760, Epoch = 19, LR = 0,000463, AvgCost = 208,5914, Sent = 1000, SentPerMin = 1142,75, WordPerSec = 1048,78
Update = 16800, Epoch = 20, LR = 0,000463, AvgCost = 204,6731, Sent = 122, SentPerMin = 1168,65, WordPerSec = 1020,01
Update = 16900, Epoch = 20, LR = 0,000462, AvgCost = 211,3147, Sent = 411, SentPerMin = 1125,79, WordPerSec = 1036,86
Update = 17000, Epoch = 20, LR = 0,000460, AvgCost = 207,5860, Sent = 709, SentPerMin = 1144,08, WordPerSec = 1045,05
Update = 17098, Epoch = 20, LR = 0,000459, AvgCost = 207,8401, Sent = 1000, SentPerMin = 1143,09, WordPerSec = 1049,09
Update = 17100, Epoch = 21, LR = 0,000459, AvgCost = 184,9042, Sent = 6, SentPerMin = 1088,27, WordPerSec = 988,51
Update = 17200, Epoch = 21, LR = 0,000457, AvgCost = 216,5372, Sent = 292, SentPerMin = 1108,23, WordPerSec = 1036,88
Update = 17300, Epoch = 21, LR = 0,000456, AvgCost = 209,5966, Sent = 595, SentPerMin = 1129,78, WordPerSec = 1039,84
Update = 17400, Epoch = 21, LR = 0,000455, AvgCost = 208,5929, Sent = 890, SentPerMin = 1135,95, WordPerSec = 1047,14
Update = 17436, Epoch = 21, LR = 0,000454, AvgCost = 207,1885, Sent = 1000, SentPerMin = 1143,20, WordPerSec = 1049,19
Update = 17500, Epoch = 22, LR = 0,000454, AvgCost = 214,1304, Sent = 184, SentPerMin = 1108,99, WordPerSec = 1018,88
Update = 17600, Epoch = 22, LR = 0,000452, AvgCost = 209,8274, Sent = 482, SentPerMin = 1127,85, WordPerSec = 1036,90
Update = 17700, Epoch = 22, LR = 0,000451, AvgCost = 209,9580, Sent = 775, SentPerMin = 1123,77, WordPerSec = 1041,19
Update = 17774, Epoch = 22, LR = 0,000450, AvgCost = 206,6368, Sent = 1000, SentPerMin = 1140,77, WordPerSec = 1046,96
Update = 17800, Epoch = 23, LR = 0,000450, AvgCost = 201,9636, Sent = 81, SentPerMin = 1145,99, WordPerSec = 1017,25
Update = 17900, Epoch = 23, LR = 0,000448, AvgCost = 211,7135, Sent = 368, SentPerMin = 1115,00, WordPerSec = 1031,88
Update = 18000, Epoch = 23, LR = 0,000447, AvgCost = 207,7025, Sent = 663, SentPerMin = 1130,39, WordPerSec = 1038,57
Update = 18100, Epoch = 23, LR = 0,000446, AvgCost = 205,0886, Sent = 969, SentPerMin = 1142,73, WordPerSec = 1044,94
Update = 18112, Epoch = 23, LR = 0,000446, AvgCost = 206,0101, Sent = 1000, SentPerMin = 1139,59, WordPerSec = 1045,87
Update = 18200, Epoch = 24, LR = 0,000445, AvgCost = 216,9996, Sent = 251, SentPerMin = 900,77, WordPerSec = 841,49
Update = 18300, Epoch = 24, LR = 0,000444, AvgCost = 207,2555, Sent = 556, SentPerMin = 930,73, WordPerSec = 851,91
Update = 18400, Epoch = 24, LR = 0,000442, AvgCost = 207,4183, Sent = 847, SentPerMin = 927,91, WordPerSec = 855,26
Update = 18450, Epoch = 24, LR = 0,000442, AvgCost = 206,0475, Sent = 1000, SentPerMin = 934,16, WordPerSec = 857,34
Update = 18500, Epoch = 25, LR = 0,000441, AvgCost = 208,9607, Sent = 145, SentPerMin = 1119,09, WordPerSec = 1017,86
Update = 18600, Epoch = 25, LR = 0,000440, AvgCost = 210,8811, Sent = 437, SentPerMin = 1119,65, WordPerSec = 1037,58
Update = 18700, Epoch = 25, LR = 0,000439, AvgCost = 205,1414, Sent = 739, SentPerMin = 1136,08, WordPerSec = 1041,43
Update = 18788, Epoch = 25, LR = 0,000438, AvgCost = 204,9779, Sent = 1000, SentPerMin = 1142,18, WordPerSec = 1048,26
Update = 18800, Epoch = 26, LR = 0,000438, AvgCost = 168,2976, Sent = 41, SentPerMin = 1271,82, WordPerSec = 979,20
Update = 18900, Epoch = 26, LR = 0,000436, AvgCost = 211,5273, Sent = 324, SentPerMin = 1116,03, WordPerSec = 1036,63
Update = 19000, Epoch = 26, LR = 0,000435, AvgCost = 205,7282, Sent = 624, SentPerMin = 1131,63, WordPerSec = 1037,42
Update = 19100, Epoch = 26, LR = 0,000434, AvgCost = 205,0548, Sent = 921, SentPerMin = 1136,98, WordPerSec = 1046,08
Update = 19126, Epoch = 26, LR = 0,000434, AvgCost = 204,3124, Sent = 1000, SentPerMin = 1141,50, WordPerSec = 1047,63
Update = 19200, Epoch = 27, LR = 0,000433, AvgCost = 209,6117, Sent = 213, SentPerMin = 1115,67, WordPerSec = 1029,25
Update = 19300, Epoch = 27, LR = 0,000432, AvgCost = 207,1151, Sent = 511, SentPerMin = 1123,11, WordPerSec = 1035,64
Update = 19400, Epoch = 27, LR = 0,000431, AvgCost = 205,2805, Sent = 808, SentPerMin = 1132,25, WordPerSec = 1042,94
Update = 19464, Epoch = 27, LR = 0,000430, AvgCost = 203,4672, Sent = 1000, SentPerMin = 1140,70, WordPerSec = 1046,89
Update = 19500, Epoch = 28, LR = 0,000430, AvgCost = 193,1467, Sent = 113, SentPerMin = 1172,73, WordPerSec = 1004,61
Update = 19600, Epoch = 28, LR = 0,000429, AvgCost = 206,0462, Sent = 399, SentPerMin = 1118,62, WordPerSec = 1029,65
Update = 19700, Epoch = 28, LR = 0,000427, AvgCost = 203,5598, Sent = 696, SentPerMin = 1134,32, WordPerSec = 1039,27
Update = 19800, Epoch = 28, LR = 0,000426, AvgCost = 202,9996, Sent = 995, SentPerMin = 1139,16, WordPerSec = 1044,77
Update = 19802, Epoch = 28, LR = 0,000426, AvgCost = 202,9501, Sent = 1000, SentPerMin = 1138,55, WordPerSec = 1044,92
Update = 19900, Epoch = 29, LR = 0,000425, AvgCost = 212,3028, Sent = 280, SentPerMin = 1105,02, WordPerSec = 1036,88
Update = 20000, Epoch = 29, LR = 0,000424, AvgCost = 203,2904, Sent = 588, SentPerMin = 1136,91, WordPerSec = 1039,97
Update = 20100, Epoch = 29, LR = 0,000423, AvgCost = 203,5710, Sent = 877, SentPerMin = 834,59, WordPerSec = 769,70
Update = 20140, Epoch = 29, LR = 0,000423, AvgCost = 202,3143, Sent = 1000, SentPerMin = 867,54, WordPerSec = 796,20
Update = 20200, Epoch = 30, LR = 0,000422, AvgCost = 204,6207, Sent = 173, SentPerMin = 1113,34, WordPerSec = 1024,31
Update = 20300, Epoch = 30, LR = 0,000421, AvgCost = 205,7797, Sent = 469, SentPerMin = 1120,14, WordPerSec = 1033,25
Update = 20400, Epoch = 30, LR = 0,000420, AvgCost = 204,2686, Sent = 766, SentPerMin = 1124,81, WordPerSec = 1037,90
Update = 20478, Epoch = 30, LR = 0,000419, AvgCost = 201,7855, Sent = 1000, SentPerMin = 1137,97, WordPerSec = 1044,39
Update = 20500, Epoch = 31, LR = 0,000419, AvgCost = 194,4931, Sent = 71, SentPerMin = 1188,49, WordPerSec = 1028,63
Update = 20600, Epoch = 31, LR = 0,000418, AvgCost = 204,6151, Sent = 358, SentPerMin = 1125,18, WordPerSec = 1033,56
Update = 20700, Epoch = 31, LR = 0,000417, AvgCost = 202,5403, Sent = 652, SentPerMin = 1133,04, WordPerSec = 1040,44
Update = 20800, Epoch = 31, LR = 0,000416, AvgCost = 201,2040, Sent = 955, SentPerMin = 1118,10, WordPerSec = 1025,78
Update = 20816, Epoch = 31, LR = 0,000416, AvgCost = 201,2487, Sent = 1000, SentPerMin = 1117,41, WordPerSec = 1025,52
Update = 20900, Epoch = 32, LR = 0,000415, AvgCost = 203,0809, Sent = 244, SentPerMin = 1096,87, WordPerSec = 1002,09
Update = 21000, Epoch = 32, LR = 0,000414, AvgCost = 202,6641, Sent = 542, SentPerMin = 1114,13, WordPerSec = 1023,27
Update = 21100, Epoch = 32, LR = 0,000413, AvgCost = 202,0972, Sent = 835, SentPerMin = 1121,11, WordPerSec = 1031,44
Update = 21154, Epoch = 32, LR = 0,000413, AvgCost = 200,9500, Sent = 1000, SentPerMin = 1130,28, WordPerSec = 1037,34
Update = 21200, Epoch = 33, LR = 0,000412, AvgCost = 198,3541, Sent = 136, SentPerMin = 1148,88, WordPerSec = 1022,31
Update = 21300, Epoch = 33, LR = 0,000411, AvgCost = 203,0876, Sent = 428, SentPerMin = 1127,59, WordPerSec = 1037,14
Update = 21400, Epoch = 33, LR = 0,000410, AvgCost = 200,7826, Sent = 725, SentPerMin = 1137,63, WordPerSec = 1042,83
Update = 21492, Epoch = 33, LR = 0,000409, AvgCost = 200,5059, Sent = 1000, SentPerMin = 1140,30, WordPerSec = 1046,53
Update = 21500, Epoch = 34, LR = 0,000409, AvgCost = 182,1136, Sent = 23, SentPerMin = 1003,13, WordPerSec = 874,47
Update = 21600, Epoch = 34, LR = 0,000408, AvgCost = 206,6855, Sent = 311, SentPerMin = 1105,97, WordPerSec = 1028,26
Update = 21700, Epoch = 34, LR = 0,000407, AvgCost = 202,4088, Sent = 611, SentPerMin = 1121,58, WordPerSec = 1033,01
Update = 21800, Epoch = 34, LR = 0,000406, AvgCost = 201,5637, Sent = 906, SentPerMin = 1128,51, WordPerSec = 1041,05
Update = 21830, Epoch = 34, LR = 0,000406, AvgCost = 200,2016, Sent = 1000, SentPerMin = 1136,74, WordPerSec = 1043,26
Update = 21900, Epoch = 35, LR = 0,000405, AvgCost = 205,5629, Sent = 200, SentPerMin = 907,40, WordPerSec = 832,69
Update = 22000, Epoch = 35, LR = 0,000405, AvgCost = 202,4506, Sent = 501, SentPerMin = 917,52, WordPerSec = 840,57
Update = 22100, Epoch = 35, LR = 0,000404, AvgCost = 202,9706, Sent = 795, SentPerMin = 918,45, WordPerSec = 846,26
Update = 22168, Epoch = 35, LR = 0,000403, AvgCost = 201,0215, Sent = 1000, SentPerMin = 926,72, WordPerSec = 850,51
Update = 22200, Epoch = 36, LR = 0,000403, AvgCost = 195,0079, Sent = 99, SentPerMin = 1166,80, WordPerSec = 1016,72
Update = 22300, Epoch = 36, LR = 0,000402, AvgCost = 201,9749, Sent = 388, SentPerMin = 1125,69, WordPerSec = 1031,73
Update = 22400, Epoch = 36, LR = 0,000401, AvgCost = 200,6103, Sent = 683, SentPerMin = 1134,18, WordPerSec = 1039,25
Update = 22500, Epoch = 36, LR = 0,000400, AvgCost = 200,1247, Sent = 984, SentPerMin = 1139,39, WordPerSec = 1045,62
Update = 22506, Epoch = 36, LR = 0,000400, AvgCost = 200,0043, Sent = 1000, SentPerMin = 1138,94, WordPerSec = 1045,28
Update = 22600, Epoch = 37, LR = 0,000399, AvgCost = 207,1210, Sent = 269, SentPerMin = 1109,69, WordPerSec = 1037,22
Update = 22700, Epoch = 37, LR = 0,000398, AvgCost = 200,6712, Sent = 573, SentPerMin = 1131,09, WordPerSec = 1038,80
Update = 22800, Epoch = 37, LR = 0,000397, AvgCost = 200,2705, Sent = 865, SentPerMin = 1130,24, WordPerSec = 1041,00
Update = 22844, Epoch = 37, LR = 0,000397, AvgCost = 199,3308, Sent = 1000, SentPerMin = 1138,30, WordPerSec = 1044,69
Update = 22900, Epoch = 38, LR = 0,000396, AvgCost = 202,3607, Sent = 161, SentPerMin = 1107,74, WordPerSec = 1016,11
Update = 23000, Epoch = 38, LR = 0,000396, AvgCost = 203,1153, Sent = 456, SentPerMin = 1118,14, WordPerSec = 1033,14
Update = 23100, Epoch = 38, LR = 0,000395, AvgCost = 199,9207, Sent = 757, SentPerMin = 1125,86, WordPerSec = 1035,58
Update = 23182, Epoch = 38, LR = 0,000394, AvgCost = 198,8137, Sent = 1000, SentPerMin = 1134,21, WordPerSec = 1040,94
Update = 23200, Epoch = 39, LR = 0,000394, AvgCost = 186,9084, Sent = 59, SentPerMin = 1227,66, WordPerSec = 1025,82
Update = 23300, Epoch = 39, LR = 0,000393, AvgCost = 202,9986, Sent = 343, SentPerMin = 1116,53, WordPerSec = 1032,33
Update = 23400, Epoch = 39, LR = 0,000392, AvgCost = 199,4105, Sent = 640, SentPerMin = 1098,61, WordPerSec = 1009,54
Update = 23500, Epoch = 39, LR = 0,000391, AvgCost = 198,3665, Sent = 943, SentPerMin = 1090,49, WordPerSec = 1000,95
Update = 23520, Epoch = 39, LR = 0,000391, AvgCost = 198,3868, Sent = 1000, SentPerMin = 1089,24, WordPerSec = 999,67
Update = 23600, Epoch = 40, LR = 0,000391, AvgCost = 200,7994, Sent = 231, SentPerMin = 1086,13, WordPerSec = 999,46
Update = 23700, Epoch = 40, LR = 0,000390, AvgCost = 200,1314, Sent = 529, SentPerMin = 1086,22, WordPerSec = 999,74
Update = 23800, Epoch = 40, LR = 0,000389, AvgCost = 199,0469, Sent = 824, SentPerMin = 1093,91, WordPerSec = 1006,60
Update = 23858, Epoch = 40, LR = 0,000388, AvgCost = 197,9053, Sent = 1000, SentPerMin = 1097,67, WordPerSec = 1007,41
Update = 23900, Epoch = 41, LR = 0,000388, AvgCost = 195,6417, Sent = 126, SentPerMin = 1106,08, WordPerSec = 976,45
Update = 24000, Epoch = 41, LR = 0,000387, AvgCost = 199,2328, Sent = 418, SentPerMin = 953,19, WordPerSec = 874,67
Update = 24100, Epoch = 41, LR = 0,000386, AvgCost = 196,9907, Sent = 715, SentPerMin = 947,57, WordPerSec = 866,55
Update = 24196, Epoch = 41, LR = 0,000386, AvgCost = 197,4840, Sent = 1000, SentPerMin = 986,78, WordPerSec = 905,64
Update = 24200, Epoch = 42, LR = 0,000386, AvgCost = 177,5696, Sent = 12, SentPerMin = 1071,82, WordPerSec = 964,64
Update = 24300, Epoch = 42, LR = 0,000385, AvgCost = 204,3863, Sent = 299, SentPerMin = 1068,20, WordPerSec = 996,40
Update = 24400, Epoch = 42, LR = 0,000384, AvgCost = 198,4222, Sent = 601, SentPerMin = 1096,45, WordPerSec = 1007,24
Update = 24500, Epoch = 42, LR = 0,000383, AvgCost = 197,8024, Sent = 897, SentPerMin = 1103,99, WordPerSec = 1015,48
Update = 24534, Epoch = 42, LR = 0,000383, AvgCost = 197,0210, Sent = 1000, SentPerMin = 1105,48, WordPerSec = 1014,57
Update = 24600, Epoch = 43, LR = 0,000383, AvgCost = 203,0550, Sent = 189, SentPerMin = 1053,78, WordPerSec = 970,71
Update = 24700, Epoch = 43, LR = 0,000382, AvgCost = 198,3340, Sent = 490, SentPerMin = 1059,31, WordPerSec = 968,41
Update = 24800, Epoch = 43, LR = 0,000381, AvgCost = 198,9274, Sent = 784, SentPerMin = 1032,84, WordPerSec = 951,88
Update = 24872, Epoch = 43, LR = 0,000380, AvgCost = 196,8353, Sent = 1000, SentPerMin = 1034,00, WordPerSec = 948,97
Update = 24900, Epoch = 44, LR = 0,000380, AvgCost = 191,8830, Sent = 87, SentPerMin = 984,73, WordPerSec = 865,32
Update = 25000, Epoch = 44, LR = 0,000379, AvgCost = 200,0000, Sent = 375, SentPerMin = 955,31, WordPerSec = 880,50
Update = 25100, Epoch = 44, LR = 0,000379, AvgCost = 198,3536, Sent = 668, SentPerMin = 1003,03, WordPerSec = 924,27
Update = 25200, Epoch = 44, LR = 0,000378, AvgCost = 195,6085, Sent = 974, SentPerMin = 1029,50, WordPerSec = 942,07
Update = 25210, Epoch = 44, LR = 0,000378, AvgCost = 196,6281, Sent = 1000, SentPerMin = 1029,19, WordPerSec = 944,55
Update = 25300, Epoch = 45, LR = 0,000377, AvgCost = 204,5316, Sent = 258, SentPerMin = 960,33, WordPerSec = 895,13
Update = 25400, Epoch = 45, LR = 0,000376, AvgCost = 197,2933, Sent = 562, SentPerMin = 983,99, WordPerSec = 902,26
Update = 25500, Epoch = 45, LR = 0,000376, AvgCost = 197,5416, Sent = 854, SentPerMin = 966,36, WordPerSec = 889,90
Update = 25548, Epoch = 45, LR = 0,000375, AvgCost = 196,4848, Sent = 1000, SentPerMin = 964,62, WordPerSec = 885,30
Update = 25600, Epoch = 46, LR = 0,000375, AvgCost = 197,9941, Sent = 152, SentPerMin = 754,81, WordPerSec = 679,08
Update = 25700, Epoch = 46, LR = 0,000374, AvgCost = 202,6185, Sent = 443, SentPerMin = 760,96, WordPerSec = 703,56
Update = 25800, Epoch = 46, LR = 0,000374, AvgCost = 198,3284, Sent = 744, SentPerMin = 777,43, WordPerSec = 714,02
Update = 25886, Epoch = 46, LR = 0,000373, AvgCost = 197,8113, Sent = 1000, SentPerMin = 783,26, WordPerSec = 718,85
Update = 25900, Epoch = 47, LR = 0,000373, AvgCost = 186,7674, Sent = 45, SentPerMin = 1156,68, WordPerSec = 947,19
Update = 26000, Epoch = 47, LR = 0,000372, AvgCost = 202,5304, Sent = 330, SentPerMin = 1050,58, WordPerSec = 973,86
Update = 26100, Epoch = 47, LR = 0,000371, AvgCost = 197,9313, Sent = 630, SentPerMin = 1058,55, WordPerSec = 970,76
Update = 26200, Epoch = 47, LR = 0,000371, AvgCost = 197,2016, Sent = 929, SentPerMin = 1067,99, WordPerSec = 981,21
Update = 26224, Epoch = 47, LR = 0,000371, AvgCost = 196,7349, Sent = 1000, SentPerMin = 1073,08, WordPerSec = 984,83
Update = 26300, Epoch = 48, LR = 0,000370, AvgCost = 200,2338, Sent = 219, SentPerMin = 1075,94, WordPerSec = 991,44
Update = 26400, Epoch = 48, LR = 0,000369, AvgCost = 198,2258, Sent = 519, SentPerMin = 1079,60, WordPerSec = 990,68
Update = 26500, Epoch = 48, LR = 0,000369, AvgCost = 197,2075, Sent = 814, SentPerMin = 1026,76, WordPerSec = 944,41
Update = 26562, Epoch = 48, LR = 0,000368, AvgCost = 195,9244, Sent = 1000, SentPerMin = 995,74, WordPerSec = 913,86
Update = 26600, Epoch = 49, LR = 0,000368, AvgCost = 182,3919, Sent = 119, SentPerMin = 899,39, WordPerSec = 765,11
Update = 26700, Epoch = 49, LR = 0,000367, AvgCost = 197,8253, Sent = 406, SentPerMin = 845,09, WordPerSec = 776,72
Update = 26800, Epoch = 49, LR = 0,000367, AvgCost = 195,5265, Sent = 702, SentPerMin = 854,09, WordPerSec = 781,09
Update = 26900, Epoch = 49, LR = 0,000366, AvgCost = 195,5077, Sent = 1000, SentPerMin = 874,01, WordPerSec = 802,14
Update = 26900, Epoch = 49, LR = 0,000366, AvgCost = 195,5077, Sent = 1000, SentPerMin = 874,00, WordPerSec = 802,13
Update = 27000, Epoch = 50, LR = 0,000365, AvgCost = 204,5261, Sent = 285, SentPerMin = 876,84, WordPerSec = 823,35
Update = 27100, Epoch = 50, LR = 0,000364, AvgCost = 196,0579, Sent = 592, SentPerMin = 889,47, WordPerSec = 815,92
Update = 27200, Epoch = 50, LR = 0,000364, AvgCost = 196,7534, Sent = 882, SentPerMin = 898,48, WordPerSec = 830,04
Update = 27238, Epoch = 50, LR = 0,000364, AvgCost = 195,0637, Sent = 1000, SentPerMin = 913,72, WordPerSec = 838,58
Update = 27300, Epoch = 51, LR = 0,000363, AvgCost = 197,2679, Sent = 178, SentPerMin = 911,37, WordPerSec = 840,97
Update = 27400, Epoch = 51, LR = 0,000362, AvgCost = 198,3190, Sent = 475, SentPerMin = 932,28, WordPerSec = 858,75
Update = 27500, Epoch = 51, LR = 0,000362, AvgCost = 197,3118, Sent = 771, SentPerMin = 922,43, WordPerSec = 852,60
Update = 27576, Epoch = 51, LR = 0,000361, AvgCost = 194,7213, Sent = 1000, SentPerMin = 930,85, WordPerSec = 854,30
Update = 27600, Epoch = 52, LR = 0,000361, AvgCost = 189,8066, Sent = 76, SentPerMin = 920,96, WordPerSec = 812,30
Update = 27700, Epoch = 52, LR = 0,000361, AvgCost = 198,4696, Sent = 363, SentPerMin = 982,78, WordPerSec = 906,25
Update = 27800, Epoch = 52, LR = 0,000360, AvgCost = 196,0623, Sent = 656, SentPerMin = 992,92, WordPerSec = 914,21
Update = 27900, Epoch = 52, LR = 0,000359, AvgCost = 193,8907, Sent = 961, SentPerMin = 970,47, WordPerSec = 889,50
Update = 27914, Epoch = 52, LR = 0,000359, AvgCost = 194,2611, Sent = 1000, SentPerMin = 968,58, WordPerSec = 888,93
Update = 28000, Epoch = 53, LR = 0,000359, AvgCost = 198,4058, Sent = 248, SentPerMin = 921,52, WordPerSec = 851,29
Update = 28100, Epoch = 53, LR = 0,000358, AvgCost = 195,4512, Sent = 549, SentPerMin = 916,61, WordPerSec = 839,84
Update = 28200, Epoch = 53, LR = 0,000357, AvgCost = 195,3950, Sent = 841, SentPerMin = 902,63, WordPerSec = 831,19
Update = 28252, Epoch = 53, LR = 0,000357, AvgCost = 194,0708, Sent = 1000, SentPerMin = 909,06, WordPerSec = 834,31
Update = 28300, Epoch = 54, LR = 0,000357, AvgCost = 192,3996, Sent = 140, SentPerMin = 1054,66, WordPerSec = 946,68
Update = 28400, Epoch = 54, LR = 0,000356, AvgCost = 199,3341, Sent = 431, SentPerMin = 1090,76, WordPerSec = 1011,09
Update = 28500, Epoch = 54, LR = 0,000355, AvgCost = 193,4075, Sent = 732, SentPerMin = 1118,80, WordPerSec = 1022,87
Update = 28590, Epoch = 54, LR = 0,000355, AvgCost = 193,7648, Sent = 1000, SentPerMin = 1124,35, WordPerSec = 1031,89
Update = 28600, Epoch = 55, LR = 0,000355, AvgCost = 160,0578, Sent = 33, SentPerMin = 1239,80, WordPerSec = 958,02
Update = 28700, Epoch = 55, LR = 0,000354, AvgCost = 199,9089, Sent = 317, SentPerMin = 1109,87, WordPerSec = 1033,55
Update = 28800, Epoch = 55, LR = 0,000354, AvgCost = 195,0735, Sent = 618, SentPerMin = 1127,12, WordPerSec = 1034,44
Update = 28900, Epoch = 55, LR = 0,000353, AvgCost = 194,5132, Sent = 914, SentPerMin = 1131,26, WordPerSec = 1042,45
Update = 28928, Epoch = 55, LR = 0,000353, AvgCost = 193,4899, Sent = 1000, SentPerMin = 1136,88, WordPerSec = 1043,39
Update = 29000, Epoch = 56, LR = 0,000352, AvgCost = 201,9928, Sent = 205, SentPerMin = 902,61, WordPerSec = 842,73
Update = 29100, Epoch = 56, LR = 0,000352, AvgCost = 196,8587, Sent = 506, SentPerMin = 922,75, WordPerSec = 846,37
Update = 29200, Epoch = 56, LR = 0,000351, AvgCost = 196,8322, Sent = 802, SentPerMin = 923,96, WordPerSec = 850,63
Update = 29266, Epoch = 56, LR = 0,000351, AvgCost = 195,0617, Sent = 1000, SentPerMin = 930,24, WordPerSec = 853,74
Update = 29300, Epoch = 57, LR = 0,000351, AvgCost = 185,3277, Sent = 106, SentPerMin = 1173,65, WordPerSec = 1010,89
Update = 29400, Epoch = 57, LR = 0,000350, AvgCost = 197,4027, Sent = 393, SentPerMin = 1118,46, WordPerSec = 1030,81
Update = 29500, Epoch = 57, LR = 0,000349, AvgCost = 194,9803, Sent = 689, SentPerMin = 1132,52, WordPerSec = 1038,14
Update = 29600, Epoch = 57, LR = 0,000349, AvgCost = 194,4184, Sent = 989, SentPerMin = 1134,96, WordPerSec = 1041,91
Update = 29604, Epoch = 57, LR = 0,000349, AvgCost = 194,2578, Sent = 1000, SentPerMin = 1134,99, WordPerSec = 1041,66
Update = 29700, Epoch = 58, LR = 0,000348, AvgCost = 203,1991, Sent = 275, SentPerMin = 1097,83, WordPerSec = 1030,57
Update = 29800, Epoch = 58, LR = 0,000348, AvgCost = 195,1114, Sent = 581, SentPerMin = 1125,84, WordPerSec = 1032,80
Update = 29900, Epoch = 58, LR = 0,000347, AvgCost = 195,0559, Sent = 870, SentPerMin = 1126,63, WordPerSec = 1040,19
Update = 29942, Epoch = 58, LR = 0,000347, AvgCost = 193,6393, Sent = 1000, SentPerMin = 1134,43, WordPerSec = 1041,14
Update = 30000, Epoch = 59, LR = 0,000346, AvgCost = 196,1221, Sent = 166, SentPerMin = 1102,24, WordPerSec = 1015,70
Update = 30100, Epoch = 59, LR = 0,000346, AvgCost = 197,1621, Sent = 463, SentPerMin = 818,32, WordPerSec = 755,31
Update = 30200, Epoch = 59, LR = 0,000345, AvgCost = 195,2879, Sent = 761, SentPerMin = 918,97, WordPerSec = 846,24
Update = 30280, Epoch = 59, LR = 0,000345, AvgCost = 193,2408, Sent = 1000, SentPerMin = 970,02, WordPerSec = 890,26
Update = 30300, Epoch = 60, LR = 0,000345, AvgCost = 189,5223, Sent = 64, SentPerMin = 1152,72, WordPerSec = 1005,02
Update = 30400, Epoch = 60, LR = 0,000344, AvgCost = 196,8308, Sent = 352, SentPerMin = 1113,26, WordPerSec = 1023,75
Update = 30500, Epoch = 60, LR = 0,000344, AvgCost = 194,6159, Sent = 645, SentPerMin = 1118,90, WordPerSec = 1029,91
Update = 30600, Epoch = 60, LR = 0,000343, AvgCost = 193,3598, Sent = 948, SentPerMin = 1128,59, WordPerSec = 1037,52
Update = 30618, Epoch = 60, LR = 0,000343, AvgCost = 192,9535, Sent = 1000, SentPerMin = 1131,09, WordPerSec = 1038,07
Update = 30700, Epoch = 61, LR = 0,000342, AvgCost = 193,8736, Sent = 239, SentPerMin = 1128,07, WordPerSec = 1030,21
Update = 30800, Epoch = 61, LR = 0,000342, AvgCost = 194,8240, Sent = 536, SentPerMin = 1119,11, WordPerSec = 1028,67
Update = 30900, Epoch = 61, LR = 0,000341, AvgCost = 193,1687, Sent = 831, SentPerMin = 1126,11, WordPerSec = 1034,03
Update = 30956, Epoch = 61, LR = 0,000341, AvgCost = 192,4502, Sent = 1000, SentPerMin = 1132,77, WordPerSec = 1039,62
Update = 31000, Epoch = 62, LR = 0,000341, AvgCost = 190,2917, Sent = 131, SentPerMin = 1142,04, WordPerSec = 1016,36
Update = 31100, Epoch = 62, LR = 0,000340, AvgCost = 194,5404, Sent = 423, SentPerMin = 1120,23, WordPerSec = 1029,48
Update = 31200, Epoch = 62, LR = 0,000340, AvgCost = 192,2647, Sent = 720, SentPerMin = 1133,89, WordPerSec = 1037,70
Update = 31294, Epoch = 62, LR = 0,000339, AvgCost = 192,2648, Sent = 1000, SentPerMin = 1133,00, WordPerSec = 1039,83
Update = 31300, Epoch = 63, LR = 0,000339, AvgCost = 191,0115, Sent = 16, SentPerMin = 1027,31, WordPerSec = 1002,70
Update = 31400, Epoch = 63, LR = 0,000339, AvgCost = 199,5064, Sent = 304, SentPerMin = 1104,75, WordPerSec = 1031,83
Update = 31500, Epoch = 63, LR = 0,000338, AvgCost = 193,7643, Sent = 606, SentPerMin = 1123,01, WordPerSec = 1032,11
Update = 31600, Epoch = 63, LR = 0,000338, AvgCost = 193,1094, Sent = 902, SentPerMin = 1114,45, WordPerSec = 1026,40
Update = 31632, Epoch = 63, LR = 0,000337, AvgCost = 192,0282, Sent = 1000, SentPerMin = 1117,19, WordPerSec = 1025,32
Update = 31700, Epoch = 64, LR = 0,000337, AvgCost = 196,0378, Sent = 195, SentPerMin = 1103,96, WordPerSec = 1014,70
Update = 31800, Epoch = 64, LR = 0,000336, AvgCost = 192,9553, Sent = 495, SentPerMin = 1126,32, WordPerSec = 1029,35
Update = 31900, Epoch = 64, LR = 0,000336, AvgCost = 193,6932, Sent = 789, SentPerMin = 1121,81, WordPerSec = 1034,39
Update = 31970, Epoch = 64, LR = 0,000336, AvgCost = 191,5349, Sent = 1000, SentPerMin = 1133,57, WordPerSec = 1040,35
Update = 32000, Epoch = 65, LR = 0,000335, AvgCost = 188,7459, Sent = 92, SentPerMin = 1141,35, WordPerSec = 1012,95
Update = 32100, Epoch = 65, LR = 0,000335, AvgCost = 194,4722, Sent = 381, SentPerMin = 1118,73, WordPerSec = 1029,47
Update = 32200, Epoch = 65, LR = 0,000334, AvgCost = 192,6223, Sent = 676, SentPerMin = 1130,02, WordPerSec = 1037,44
Update = 32300, Epoch = 65, LR = 0,000334, AvgCost = 190,3365, Sent = 979, SentPerMin = 1139,51, WordPerSec = 1043,40
Update = 32308, Epoch = 65, LR = 0,000334, AvgCost = 191,3446, Sent = 1000, SentPerMin = 1137,10, WordPerSec = 1043,59
Update = 32400, Epoch = 66, LR = 0,000333, AvgCost = 197,2598, Sent = 265, SentPerMin = 1113,13, WordPerSec = 1033,04
Update = 32500, Epoch = 66, LR = 0,000333, AvgCost = 193,1137, Sent = 566, SentPerMin = 1125,95, WordPerSec = 1036,30
Update = 32600, Epoch = 66, LR = 0,000332, AvgCost = 192,3024, Sent = 859, SentPerMin = 1126,59, WordPerSec = 1038,15
Update = 32646, Epoch = 66, LR = 0,000332, AvgCost = 191,1502, Sent = 1000, SentPerMin = 1135,74, WordPerSec = 1042,35
Update = 32700, Epoch = 67, LR = 0,000332, AvgCost = 194,3623, Sent = 157, SentPerMin = 917,84, WordPerSec = 834,92
Update = 32800, Epoch = 67, LR = 0,000331, AvgCost = 197,1319, Sent = 450, SentPerMin = 917,07, WordPerSec = 845,64
Update = 32900, Epoch = 67, LR = 0,000331, AvgCost = 193,0721, Sent = 752, SentPerMin = 925,64, WordPerSec = 848,64
Update = 32984, Epoch = 67, LR = 0,000330, AvgCost = 193,0080, Sent = 1000, SentPerMin = 929,88, WordPerSec = 853,41
Update = 33000, Epoch = 68, LR = 0,000330, AvgCost = 187,4562, Sent = 51, SentPerMin = 1194,28, WordPerSec = 1029,58
Update = 33100, Epoch = 68, LR = 0,000330, AvgCost = 197,5321, Sent = 336, SentPerMin = 1112,39, WordPerSec = 1030,62
Update = 33200, Epoch = 68, LR = 0,000329, AvgCost = 192,8662, Sent = 636, SentPerMin = 1125,92, WordPerSec = 1031,59
Update = 33300, Epoch = 68, LR = 0,000329, AvgCost = 192,5291, Sent = 935, SentPerMin = 1131,29, WordPerSec = 1039,94
Update = 33322, Epoch = 68, LR = 0,000329, AvgCost = 192,1464, Sent = 1000, SentPerMin = 1135,60, WordPerSec = 1042,22
Update = 33400, Epoch = 69, LR = 0,000328, AvgCost = 194,2438, Sent = 225, SentPerMin = 1122,82, WordPerSec = 1032,16
Update = 33500, Epoch = 69, LR = 0,000328, AvgCost = 194,4231, Sent = 524, SentPerMin = 1121,29, WordPerSec = 1031,74
Update = 33600, Epoch = 69, LR = 0,000327, AvgCost = 192,7089, Sent = 819, SentPerMin = 1126,31, WordPerSec = 1035,98
Update = 33660, Epoch = 69, LR = 0,000327, AvgCost = 191,5905, Sent = 1000, SentPerMin = 1134,46, WordPerSec = 1041,17
Update = 33700, Epoch = 70, LR = 0,000327, AvgCost = 187,2102, Sent = 122, SentPerMin = 1157,92, WordPerSec = 1010,65
Update = 33800, Epoch = 70, LR = 0,000326, AvgCost = 194,4382, Sent = 411, SentPerMin = 1115,91, WordPerSec = 1027,75
Update = 33900, Epoch = 70, LR = 0,000326, AvgCost = 191,1285, Sent = 709, SentPerMin = 1134,84, WordPerSec = 1036,61
Update = 33998, Epoch = 70, LR = 0,000325, AvgCost = 191,3228, Sent = 1000, SentPerMin = 1133,98, WordPerSec = 1040,73
Update = 34000, Epoch = 71, LR = 0,000325, AvgCost = 168,1615, Sent = 6, SentPerMin = 1046,06, WordPerSec = 950,17
Update = 34100, Epoch = 71, LR = 0,000325, AvgCost = 199,0423, Sent = 292, SentPerMin = 1096,57, WordPerSec = 1025,96
Update = 34200, Epoch = 71, LR = 0,000324, AvgCost = 193,1600, Sent = 595, SentPerMin = 1120,67, WordPerSec = 1031,46
Update = 34300, Epoch = 71, LR = 0,000324, AvgCost = 192,1449, Sent = 890, SentPerMin = 1125,61, WordPerSec = 1037,60
Update = 34336, Epoch = 71, LR = 0,000324, AvgCost = 190,9171, Sent = 1000, SentPerMin = 1132,07, WordPerSec = 1038,97
Update = 34400, Epoch = 72, LR = 0,000323, AvgCost = 196,5788, Sent = 184, SentPerMin = 1110,70, WordPerSec = 1020,45
Update = 34500, Epoch = 72, LR = 0,000323, AvgCost = 193,8110, Sent = 482, SentPerMin = 1124,54, WordPerSec = 1033,86
Update = 34600, Epoch = 72, LR = 0,000323, AvgCost = 193,7435, Sent = 775, SentPerMin = 1120,95, WordPerSec = 1038,58
Update = 34674, Epoch = 72, LR = 0,000322, AvgCost = 190,7565, Sent = 1000, SentPerMin = 1135,87, WordPerSec = 1042,47
Starting inference...
Inference results:
在在
和了了 在在   和 在 ) 的 ) 的 ) 的。
和了了 了 在 在 在 在 的 ) 的 的  在 在 在    在
在和必要
。

             */

            /*
             Update = 10100, Epoch = 0, LR = 0,000597, AvgCost = 265,8297, Sent = 285, SentPerMin = 843,90, WordPerSec = 792,43
Update = 10200, Epoch = 0, LR = 0,000594, AvgCost = 247,1876, Sent = 592, SentPerMin = 893,71, WordPerSec = 819,82
Update = 10300, Epoch = 0, LR = 0,000591, AvgCost = 244,1274, Sent = 882, SentPerMin = 900,79, WordPerSec = 832,17
Update = 10338, Epoch = 0, LR = 0,000590, AvgCost = 241,0340, Sent = 1000, SentPerMin = 910,45, WordPerSec = 835,58
Update = 10400, Epoch = 1, LR = 0,000588, AvgCost = 233,2838, Sent = 178, SentPerMin = 1097,30, WordPerSec = 1012,54
Update = 10500, Epoch = 1, LR = 0,000586, AvgCost = 233,3956, Sent = 475, SentPerMin = 1114,48, WordPerSec = 1026,58
Update = 10600, Epoch = 1, LR = 0,000583, AvgCost = 231,7586, Sent = 771, SentPerMin = 1116,33, WordPerSec = 1031,82
Update = 10676, Epoch = 1, LR = 0,000581, AvgCost = 228,5596, Sent = 1000, SentPerMin = 1129,44, WordPerSec = 1036,56
Update = 10700, Epoch = 2, LR = 0,000580, AvgCost = 221,1387, Sent = 76, SentPerMin = 1136,42, WordPerSec = 1002,34
Update = 10800, Epoch = 2, LR = 0,000577, AvgCost = 230,5559, Sent = 363, SentPerMin = 1108,17, WordPerSec = 1021,88
Update = 10900, Epoch = 2, LR = 0,000575, AvgCost = 227,0580, Sent = 656, SentPerMin = 1116,92, WordPerSec = 1028,39
Update = 11000, Epoch = 2, LR = 0,000572, AvgCost = 224,5340, Sent = 961, SentPerMin = 1128,55, WordPerSec = 1034,38
Update = 11014, Epoch = 2, LR = 0,000572, AvgCost = 224,9210, Sent = 1000, SentPerMin = 1128,47, WordPerSec = 1035,67
Update = 11100, Epoch = 3, LR = 0,000569, AvgCost = 228,2705, Sent = 248, SentPerMin = 1106,44, WordPerSec = 1022,12
Update = 11200, Epoch = 3, LR = 0,000567, AvgCost = 224,0952, Sent = 549, SentPerMin = 1121,78, WordPerSec = 1027,82
Update = 11300, Epoch = 3, LR = 0,000564, AvgCost = 224,0849, Sent = 841, SentPerMin = 1119,74, WordPerSec = 1031,11
Update = 11352, Epoch = 3, LR = 0,000563, AvgCost = 222,6678, Sent = 1000, SentPerMin = 1127,84, WordPerSec = 1035,09
Update = 11400, Epoch = 4, LR = 0,000562, AvgCost = 220,2284, Sent = 140, SentPerMin = 1124,84, WordPerSec = 1009,67
Update = 11500, Epoch = 4, LR = 0,000560, AvgCost = 227,2974, Sent = 431, SentPerMin = 1103,33, WordPerSec = 1022,74
Update = 11600, Epoch = 4, LR = 0,000557, AvgCost = 220,3769, Sent = 732, SentPerMin = 1123,32, WordPerSec = 1027,00
Update = 11690, Epoch = 4, LR = 0,000555, AvgCost = 220,7609, Sent = 1000, SentPerMin = 1124,79, WordPerSec = 1032,29
Update = 11700, Epoch = 5, LR = 0,000555, AvgCost = 181,9398, Sent = 33, SentPerMin = 1244,61, WordPerSec = 961,74
Update = 11800, Epoch = 5, LR = 0,000552, AvgCost = 226,6427, Sent = 317, SentPerMin = 1094,64, WordPerSec = 1019,36
Update = 11900, Epoch = 5, LR = 0,000550, AvgCost = 220,4357, Sent = 618, SentPerMin = 1115,05, WordPerSec = 1023,36
Update = 12000, Epoch = 5, LR = 0,000548, AvgCost = 220,1355, Sent = 914, SentPerMin = 1117,68, WordPerSec = 1029,94
Update = 12028, Epoch = 5, LR = 0,000547, AvgCost = 219,0244, Sent = 1000, SentPerMin = 1124,45, WordPerSec = 1031,98
Update = 12100, Epoch = 6, LR = 0,000545, AvgCost = 226,8588, Sent = 205, SentPerMin = 1083,24, WordPerSec = 1011,37
Update = 12200, Epoch = 6, LR = 0,000543, AvgCost = 219,6432, Sent = 506, SentPerMin = 1112,99, WordPerSec = 1020,86
Update = 12300, Epoch = 6, LR = 0,000541, AvgCost = 219,7699, Sent = 802, SentPerMin = 1116,56, WordPerSec = 1027,95
Update = 12366, Epoch = 6, LR = 0,000540, AvgCost = 217,8765, Sent = 1000, SentPerMin = 1126,43, WordPerSec = 1033,80
Update = 12400, Epoch = 7, LR = 0,000539, AvgCost = 207,5244, Sent = 106, SentPerMin = 1152,87, WordPerSec = 992,99
Update = 12500, Epoch = 7, LR = 0,000537, AvgCost = 220,0732, Sent = 393, SentPerMin = 1102,15, WordPerSec = 1015,77
Update = 12600, Epoch = 7, LR = 0,000535, AvgCost = 217,0312, Sent = 689, SentPerMin = 1120,38, WordPerSec = 1027,01
Update = 12700, Epoch = 7, LR = 0,000532, AvgCost = 216,8038, Sent = 989, SentPerMin = 1124,63, WordPerSec = 1032,42
Update = 12704, Epoch = 7, LR = 0,000532, AvgCost = 216,6559, Sent = 1000, SentPerMin = 1124,23, WordPerSec = 1031,78
Update = 12800, Epoch = 8, LR = 0,000530, AvgCost = 226,4330, Sent = 275, SentPerMin = 1087,86, WordPerSec = 1021,21
Update = 12900, Epoch = 8, LR = 0,000528, AvgCost = 216,6930, Sent = 581, SentPerMin = 1115,96, WordPerSec = 1023,73
Update = 13000, Epoch = 8, LR = 0,000526, AvgCost = 217,2705, Sent = 870, SentPerMin = 1114,73, WordPerSec = 1029,20
Update = 13042, Epoch = 8, LR = 0,000525, AvgCost = 215,6988, Sent = 1000, SentPerMin = 1123,68, WordPerSec = 1031,28
Update = 13100, Epoch = 9, LR = 0,000524, AvgCost = 218,8900, Sent = 166, SentPerMin = 1086,66, WordPerSec = 1001,34
Update = 13200, Epoch = 9, LR = 0,000522, AvgCost = 218,8484, Sent = 463, SentPerMin = 1102,63, WordPerSec = 1017,73
Update = 13300, Epoch = 9, LR = 0,000520, AvgCost = 217,0254, Sent = 761, SentPerMin = 1108,80, WordPerSec = 1021,04
Update = 13380, Epoch = 9, LR = 0,000519, AvgCost = 214,9791, Sent = 1000, SentPerMin = 1121,99, WordPerSec = 1029,73
Update = 13400, Epoch = 10, LR = 0,000518, AvgCost = 208,0534, Sent = 64, SentPerMin = 1169,95, WordPerSec = 1020,05
Update = 13500, Epoch = 10, LR = 0,000516, AvgCost = 218,1338, Sent = 352, SentPerMin = 1110,19, WordPerSec = 1020,93
Update = 13600, Epoch = 10, LR = 0,000514, AvgCost = 215,5762, Sent = 645, SentPerMin = 1112,42, WordPerSec = 1023,95
Update = 13700, Epoch = 10, LR = 0,000513, AvgCost = 214,7220, Sent = 948, SentPerMin = 1113,17, WordPerSec = 1023,34
Update = 13718, Epoch = 10, LR = 0,000512, AvgCost = 214,1842, Sent = 1000, SentPerMin = 1115,12, WordPerSec = 1023,42
Update = 13800, Epoch = 11, LR = 0,000511, AvgCost = 215,6371, Sent = 239, SentPerMin = 1119,24, WordPerSec = 1022,15
Update = 13900, Epoch = 11, LR = 0,000509, AvgCost = 215,6103, Sent = 536, SentPerMin = 1114,51, WordPerSec = 1024,44
Update = 14000, Epoch = 11, LR = 0,000507, AvgCost = 214,3830, Sent = 831, SentPerMin = 1118,71, WordPerSec = 1027,24
Update = 14056, Epoch = 11, LR = 0,000506, AvgCost = 213,5401, Sent = 1000, SentPerMin = 1124,21, WordPerSec = 1031,76
Update = 14100, Epoch = 12, LR = 0,000505, AvgCost = 212,2673, Sent = 131, SentPerMin = 925,32, WordPerSec = 823,49
Update = 14200, Epoch = 12, LR = 0,000504, AvgCost = 215,5954, Sent = 423, SentPerMin = 910,60, WordPerSec = 836,83
Update = 14300, Epoch = 12, LR = 0,000502, AvgCost = 213,1946, Sent = 720, SentPerMin = 920,91, WordPerSec = 842,78
Update = 14394, Epoch = 12, LR = 0,000500, AvgCost = 213,2704, Sent = 1000, SentPerMin = 922,30, WordPerSec = 846,46
Update = 14400, Epoch = 13, LR = 0,000500, AvgCost = 215,4689, Sent = 16, SentPerMin = 1016,40, WordPerSec = 992,05
Update = 14500, Epoch = 13, LR = 0,000498, AvgCost = 220,9730, Sent = 304, SentPerMin = 1097,78, WordPerSec = 1025,31
Update = 14600, Epoch = 13, LR = 0,000497, AvgCost = 214,2906, Sent = 606, SentPerMin = 1116,21, WordPerSec = 1025,87
Update = 14700, Epoch = 13, LR = 0,000495, AvgCost = 214,0214, Sent = 902, SentPerMin = 1120,20, WordPerSec = 1031,69
Update = 14732, Epoch = 13, LR = 0,000494, AvgCost = 212,7121, Sent = 1000, SentPerMin = 1126,66, WordPerSec = 1034,01
Update = 14800, Epoch = 14, LR = 0,000493, AvgCost = 218,0194, Sent = 195, SentPerMin = 1097,72, WordPerSec = 1008,97
Update = 14900, Epoch = 14, LR = 0,000492, AvgCost = 213,4304, Sent = 495, SentPerMin = 1119,15, WordPerSec = 1022,80
Update = 15000, Epoch = 14, LR = 0,000490, AvgCost = 214,5305, Sent = 789, SentPerMin = 1112,82, WordPerSec = 1026,10
Update = 15070, Epoch = 14, LR = 0,000489, AvgCost = 212,1388, Sent = 1000, SentPerMin = 1124,32, WordPerSec = 1031,86
Update = 15100, Epoch = 15, LR = 0,000488, AvgCost = 209,9155, Sent = 92, SentPerMin = 1127,34, WordPerSec = 1000,52
Update = 15200, Epoch = 15, LR = 0,000487, AvgCost = 214,6098, Sent = 381, SentPerMin = 1106,36, WordPerSec = 1018,08
Update = 15300, Epoch = 15, LR = 0,000485, AvgCost = 212,4273, Sent = 676, SentPerMin = 1119,22, WordPerSec = 1027,53
Update = 15400, Epoch = 15, LR = 0,000483, AvgCost = 210,3008, Sent = 979, SentPerMin = 1129,33, WordPerSec = 1034,09
Update = 15408, Epoch = 15, LR = 0,000483, AvgCost = 211,3672, Sent = 1000, SentPerMin = 1126,93, WordPerSec = 1034,26
Update = 15500, Epoch = 16, LR = 0,000482, AvgCost = 218,2121, Sent = 265, SentPerMin = 1101,20, WordPerSec = 1021,97
Update = 15600, Epoch = 16, LR = 0,000480, AvgCost = 212,5334, Sent = 566, SentPerMin = 1114,73, WordPerSec = 1025,97
Update = 15700, Epoch = 16, LR = 0,000479, AvgCost = 212,0421, Sent = 859, SentPerMin = 1116,73, WordPerSec = 1029,07
Update = 15746, Epoch = 16, LR = 0,000478, AvgCost = 210,6957, Sent = 1000, SentPerMin = 1126,33, WordPerSec = 1033,71
Update = 15800, Epoch = 17, LR = 0,000477, AvgCost = 212,9957, Sent = 157, SentPerMin = 1109,26, WordPerSec = 1009,05
Update = 15900, Epoch = 17, LR = 0,000476, AvgCost = 214,3993, Sent = 450, SentPerMin = 1106,99, WordPerSec = 1020,77
Update = 16000, Epoch = 17, LR = 0,000474, AvgCost = 210,1797, Sent = 752, SentPerMin = 1119,49, WordPerSec = 1026,37
Update = 16084, Epoch = 17, LR = 0,000473, AvgCost = 210,1937, Sent = 1000, SentPerMin = 1126,82, WordPerSec = 1034,16
Update = 16100, Epoch = 18, LR = 0,000473, AvgCost = 204,4726, Sent = 51, SentPerMin = 1178,59, WordPerSec = 1016,05
Update = 16200, Epoch = 18, LR = 0,000471, AvgCost = 215,5806, Sent = 336, SentPerMin = 1101,99, WordPerSec = 1020,98
Update = 16300, Epoch = 18, LR = 0,000470, AvgCost = 210,1503, Sent = 636, SentPerMin = 1117,31, WordPerSec = 1023,71
Update = 16400, Epoch = 18, LR = 0,000469, AvgCost = 210,0277, Sent = 935, SentPerMin = 1121,99, WordPerSec = 1031,39
Update = 16422, Epoch = 18, LR = 0,000468, AvgCost = 209,6191, Sent = 1000, SentPerMin = 1126,26, WordPerSec = 1033,65
Update = 16500, Epoch = 19, LR = 0,000467, AvgCost = 213,3919, Sent = 225, SentPerMin = 1106,62, WordPerSec = 1017,27
Update = 16600, Epoch = 19, LR = 0,000466, AvgCost = 211,9631, Sent = 524, SentPerMin = 1113,37, WordPerSec = 1024,45
Update = 16700, Epoch = 19, LR = 0,000464, AvgCost = 210,3454, Sent = 819, SentPerMin = 1117,17, WordPerSec = 1027,58
Update = 16760, Epoch = 19, LR = 0,000463, AvgCost = 209,0773, Sent = 1000, SentPerMin = 1124,79, WordPerSec = 1032,30
Update = 16800, Epoch = 20, LR = 0,000463, AvgCost = 205,5031, Sent = 122, SentPerMin = 1156,29, WordPerSec = 1009,23
Update = 16900, Epoch = 20, LR = 0,000462, AvgCost = 211,6683, Sent = 411, SentPerMin = 1109,91, WordPerSec = 1022,23
Update = 17000, Epoch = 20, LR = 0,000460, AvgCost = 208,0709, Sent = 709, SentPerMin = 1127,77, WordPerSec = 1030,16
Update = 17098, Epoch = 20, LR = 0,000459, AvgCost = 208,4249, Sent = 1000, SentPerMin = 1124,20, WordPerSec = 1031,75
Update = 17100, Epoch = 21, LR = 0,000459, AvgCost = 184,9868, Sent = 6, SentPerMin = 1090,22, WordPerSec = 990,29
Update = 17200, Epoch = 21, LR = 0,000457, AvgCost = 217,6568, Sent = 292, SentPerMin = 1093,77, WordPerSec = 1023,35
Update = 17300, Epoch = 21, LR = 0,000456, AvgCost = 210,2388, Sent = 595, SentPerMin = 1113,42, WordPerSec = 1024,78
Update = 17400, Epoch = 21, LR = 0,000455, AvgCost = 209,5040, Sent = 890, SentPerMin = 1115,94, WordPerSec = 1028,69
Update = 17436, Epoch = 21, LR = 0,000454, AvgCost = 208,0816, Sent = 1000, SentPerMin = 1123,59, WordPerSec = 1031,20
Update = 17500, Epoch = 22, LR = 0,000454, AvgCost = 214,6056, Sent = 184, SentPerMin = 1099,95, WordPerSec = 1010,58
Update = 17600, Epoch = 22, LR = 0,000452, AvgCost = 210,4115, Sent = 482, SentPerMin = 1107,12, WordPerSec = 1017,84
Update = 17700, Epoch = 22, LR = 0,000451, AvgCost = 210,8386, Sent = 775, SentPerMin = 1101,23, WordPerSec = 1020,31
Update = 17774, Epoch = 22, LR = 0,000450, AvgCost = 207,5592, Sent = 1000, SentPerMin = 1118,47, WordPerSec = 1026,49
Update = 17800, Epoch = 23, LR = 0,000450, AvgCost = 202,3497, Sent = 81, SentPerMin = 1135,12, WordPerSec = 1007,59
Update = 17900, Epoch = 23, LR = 0,000448, AvgCost = 212,3275, Sent = 368, SentPerMin = 1098,50, WordPerSec = 1016,61
Update = 18000, Epoch = 23, LR = 0,000447, AvgCost = 208,3211, Sent = 663, SentPerMin = 1113,26, WordPerSec = 1022,84
Update = 18100, Epoch = 23, LR = 0,000446, AvgCost = 206,0084, Sent = 969, SentPerMin = 1126,17, WordPerSec = 1029,81
Update = 18112, Epoch = 23, LR = 0,000446, AvgCost = 206,9566, Sent = 1000, SentPerMin = 1122,84, WordPerSec = 1030,51
Update = 18200, Epoch = 24, LR = 0,000445, AvgCost = 217,3536, Sent = 251, SentPerMin = 899,93, WordPerSec = 840,71
Update = 18300, Epoch = 24, LR = 0,000444, AvgCost = 207,9318, Sent = 556, SentPerMin = 921,42, WordPerSec = 843,40
Update = 18400, Epoch = 24, LR = 0,000442, AvgCost = 208,2273, Sent = 847, SentPerMin = 916,30, WordPerSec = 844,56
Update = 18450, Epoch = 24, LR = 0,000442, AvgCost = 206,8613, Sent = 1000, SentPerMin = 923,41, WordPerSec = 847,47
Update = 18500, Epoch = 25, LR = 0,000441, AvgCost = 209,1970, Sent = 145, SentPerMin = 1099,29, WordPerSec = 999,85
Update = 18600, Epoch = 25, LR = 0,000440, AvgCost = 211,6332, Sent = 437, SentPerMin = 1104,21, WordPerSec = 1023,27
Update = 18700, Epoch = 25, LR = 0,000439, AvgCost = 205,9542, Sent = 739, SentPerMin = 1118,51, WordPerSec = 1025,32
Update = 18788, Epoch = 25, LR = 0,000438, AvgCost = 205,8933, Sent = 1000, SentPerMin = 1122,74, WordPerSec = 1030,41
Update = 18800, Epoch = 26, LR = 0,000438, AvgCost = 168,3169, Sent = 41, SentPerMin = 1249,94, WordPerSec = 962,35
Update = 18900, Epoch = 26, LR = 0,000436, AvgCost = 212,1444, Sent = 324, SentPerMin = 1098,73, WordPerSec = 1020,56
Update = 19000, Epoch = 26, LR = 0,000435, AvgCost = 206,4300, Sent = 624, SentPerMin = 1114,25, WordPerSec = 1021,49
Update = 19100, Epoch = 26, LR = 0,000434, AvgCost = 205,9261, Sent = 921, SentPerMin = 1119,27, WordPerSec = 1029,79
Update = 19126, Epoch = 26, LR = 0,000434, AvgCost = 205,1092, Sent = 1000, SentPerMin = 1123,88, WordPerSec = 1031,46
Update = 19200, Epoch = 27, LR = 0,000433, AvgCost = 210,5477, Sent = 213, SentPerMin = 1101,11, WordPerSec = 1015,82
Update = 19300, Epoch = 27, LR = 0,000432, AvgCost = 208,0103, Sent = 511, SentPerMin = 1106,55, WordPerSec = 1020,37
Update = 19400, Epoch = 27, LR = 0,000431, AvgCost = 206,3614, Sent = 808, SentPerMin = 1115,41, WordPerSec = 1027,43
Update = 19464, Epoch = 27, LR = 0,000430, AvgCost = 204,4234, Sent = 1000, SentPerMin = 1123,16, WordPerSec = 1030,80
Update = 19500, Epoch = 28, LR = 0,000430, AvgCost = 193,9193, Sent = 113, SentPerMin = 1164,10, WordPerSec = 997,21
Update = 19600, Epoch = 28, LR = 0,000429, AvgCost = 207,1358, Sent = 399, SentPerMin = 1101,60, WordPerSec = 1013,99
Update = 19700, Epoch = 28, LR = 0,000427, AvgCost = 204,5096, Sent = 696, SentPerMin = 1118,69, WordPerSec = 1024,96
Update = 19800, Epoch = 28, LR = 0,000426, AvgCost = 203,9516, Sent = 995, SentPerMin = 1123,98, WordPerSec = 1030,84
Update = 19802, Epoch = 28, LR = 0,000426, AvgCost = 203,8960, Sent = 1000, SentPerMin = 1123,24, WordPerSec = 1030,87
Update = 19900, Epoch = 29, LR = 0,000425, AvgCost = 213,5584, Sent = 280, SentPerMin = 1086,96, WordPerSec = 1019,93
Update = 20000, Epoch = 29, LR = 0,000424, AvgCost = 204,1660, Sent = 588, SentPerMin = 1118,07, WordPerSec = 1022,74
Update = 20100, Epoch = 29, LR = 0,000423, AvgCost = 204,6989, Sent = 877, SentPerMin = 868,74, WordPerSec = 801,19
Update = 20140, Epoch = 29, LR = 0,000423, AvgCost = 203,3484, Sent = 1000, SentPerMin = 898,19, WordPerSec = 824,33
Update = 20200, Epoch = 30, LR = 0,000422, AvgCost = 206,0261, Sent = 173, SentPerMin = 1091,36, WordPerSec = 1004,10
Update = 20300, Epoch = 30, LR = 0,000421, AvgCost = 206,8948, Sent = 469, SentPerMin = 1101,37, WordPerSec = 1015,93
Update = 20400, Epoch = 30, LR = 0,000420, AvgCost = 205,4442, Sent = 766, SentPerMin = 1106,89, WordPerSec = 1021,37
Update = 20478, Epoch = 30, LR = 0,000419, AvgCost = 202,8316, Sent = 1000, SentPerMin = 1120,65, WordPerSec = 1028,50
Update = 20500, Epoch = 31, LR = 0,000419, AvgCost = 196,0974, Sent = 71, SentPerMin = 1161,99, WordPerSec = 1005,70
Update = 20600, Epoch = 31, LR = 0,000418, AvgCost = 206,6978, Sent = 358, SentPerMin = 1102,95, WordPerSec = 1013,14
Update = 20700, Epoch = 31, LR = 0,000417, AvgCost = 203,9292, Sent = 652, SentPerMin = 1110,40, WordPerSec = 1019,65
Update = 20800, Epoch = 31, LR = 0,000416, AvgCost = 202,3845, Sent = 955, SentPerMin = 1117,78, WordPerSec = 1025,49
Update = 20816, Epoch = 31, LR = 0,000416, AvgCost = 202,3426, Sent = 1000, SentPerMin = 1118,17, WordPerSec = 1026,22
Update = 20900, Epoch = 32, LR = 0,000415, AvgCost = 204,9286, Sent = 244, SentPerMin = 1106,18, WordPerSec = 1010,59
Update = 21000, Epoch = 32, LR = 0,000414, AvgCost = 204,0920, Sent = 542, SentPerMin = 1108,77, WordPerSec = 1018,35
Update = 21100, Epoch = 32, LR = 0,000413, AvgCost = 203,3308, Sent = 835, SentPerMin = 1111,37, WordPerSec = 1022,48
Update = 21154, Epoch = 32, LR = 0,000413, AvgCost = 201,8682, Sent = 1000, SentPerMin = 1118,99, WordPerSec = 1026,97
Update = 21200, Epoch = 33, LR = 0,000412, AvgCost = 200,6536, Sent = 136, SentPerMin = 1127,44, WordPerSec = 1003,23
Update = 21300, Epoch = 33, LR = 0,000411, AvgCost = 204,5180, Sent = 428, SentPerMin = 1105,03, WordPerSec = 1016,38
Update = 21400, Epoch = 33, LR = 0,000410, AvgCost = 201,9005, Sent = 725, SentPerMin = 1115,61, WordPerSec = 1022,64
Update = 21492, Epoch = 33, LR = 0,000409, AvgCost = 201,4213, Sent = 1000, SentPerMin = 1119,28, WordPerSec = 1027,24
Update = 21500, Epoch = 34, LR = 0,000409, AvgCost = 183,8844, Sent = 23, SentPerMin = 1052,85, WordPerSec = 917,81
Update = 21600, Epoch = 34, LR = 0,000408, AvgCost = 208,5099, Sent = 311, SentPerMin = 1090,91, WordPerSec = 1014,27
Update = 21700, Epoch = 34, LR = 0,000407, AvgCost = 203,3897, Sent = 611, SentPerMin = 1104,78, WordPerSec = 1017,53
Update = 21800, Epoch = 34, LR = 0,000406, AvgCost = 202,6273, Sent = 906, SentPerMin = 1110,33, WordPerSec = 1024,28
Update = 21830, Epoch = 34, LR = 0,000406, AvgCost = 201,0780, Sent = 1000, SentPerMin = 1118,05, WordPerSec = 1026,11
Update = 21900, Epoch = 35, LR = 0,000405, AvgCost = 207,2603, Sent = 200, SentPerMin = 892,59, WordPerSec = 819,10
Update = 22000, Epoch = 35, LR = 0,000405, AvgCost = 203,7259, Sent = 501, SentPerMin = 906,94, WordPerSec = 830,88
Update = 22100, Epoch = 35, LR = 0,000404, AvgCost = 203,8688, Sent = 795, SentPerMin = 904,20, WordPerSec = 833,13
Update = 22168, Epoch = 35, LR = 0,000403, AvgCost = 201,6056, Sent = 1000, SentPerMin = 912,36, WordPerSec = 837,33
Update = 22200, Epoch = 36, LR = 0,000403, AvgCost = 197,2662, Sent = 99, SentPerMin = 1146,12, WordPerSec = 998,70
Update = 22300, Epoch = 36, LR = 0,000402, AvgCost = 203,5477, Sent = 388, SentPerMin = 1104,80, WordPerSec = 1012,59
Update = 22400, Epoch = 36, LR = 0,000401, AvgCost = 201,5884, Sent = 683, SentPerMin = 1114,16, WordPerSec = 1020,90
Update = 22500, Epoch = 36, LR = 0,000400, AvgCost = 200,8044, Sent = 984, SentPerMin = 1115,90, WordPerSec = 1024,06
Update = 22506, Epoch = 36, LR = 0,000400, AvgCost = 200,6666, Sent = 1000, SentPerMin = 1114,91, WordPerSec = 1023,23
Update = 22600, Epoch = 37, LR = 0,000399, AvgCost = 208,9085, Sent = 269, SentPerMin = 1089,40, WordPerSec = 1018,26
Update = 22700, Epoch = 37, LR = 0,000398, AvgCost = 201,8652, Sent = 573, SentPerMin = 1110,81, WordPerSec = 1020,18
Update = 22800, Epoch = 37, LR = 0,000397, AvgCost = 201,0495, Sent = 865, SentPerMin = 1109,50, WordPerSec = 1021,89
Update = 22844, Epoch = 37, LR = 0,000397, AvgCost = 199,8874, Sent = 1000, SentPerMin = 1116,10, WordPerSec = 1024,32
Update = 22900, Epoch = 38, LR = 0,000396, AvgCost = 204,3098, Sent = 161, SentPerMin = 1073,92, WordPerSec = 985,09
Update = 23000, Epoch = 38, LR = 0,000396, AvgCost = 204,4457, Sent = 456, SentPerMin = 1003,12, WordPerSec = 926,86
Update = 23100, Epoch = 38, LR = 0,000395, AvgCost = 200,7786, Sent = 757, SentPerMin = 1019,25, WordPerSec = 937,53
Update = 23182, Epoch = 38, LR = 0,000394, AvgCost = 199,4062, Sent = 1000, SentPerMin = 1028,05, WordPerSec = 943,51
Update = 23200, Epoch = 39, LR = 0,000394, AvgCost = 187,7962, Sent = 59, SentPerMin = 1112,36, WordPerSec = 929,48
Update = 23300, Epoch = 39, LR = 0,000393, AvgCost = 204,1822, Sent = 343, SentPerMin = 993,37, WordPerSec = 918,46
Update = 23400, Epoch = 39, LR = 0,000392, AvgCost = 200,3314, Sent = 640, SentPerMin = 1003,69, WordPerSec = 922,32
Update = 23500, Epoch = 39, LR = 0,000391, AvgCost = 199,1771, Sent = 943, SentPerMin = 1015,28, WordPerSec = 931,91
Update = 23520, Epoch = 39, LR = 0,000391, AvgCost = 199,0569, Sent = 1000, SentPerMin = 1016,80, WordPerSec = 933,19
Update = 23600, Epoch = 40, LR = 0,000391, AvgCost = 202,4608, Sent = 231, SentPerMin = 1006,50, WordPerSec = 926,19
Update = 23700, Epoch = 40, LR = 0,000390, AvgCost = 201,5699, Sent = 529, SentPerMin = 1009,53, WordPerSec = 929,16
Update = 23800, Epoch = 40, LR = 0,000389, AvgCost = 200,1895, Sent = 824, SentPerMin = 1007,90, WordPerSec = 927,46
Update = 23858, Epoch = 40, LR = 0,000388, AvgCost = 198,7559, Sent = 1000, SentPerMin = 1014,90, WordPerSec = 931,44
Update = 23900, Epoch = 41, LR = 0,000388, AvgCost = 196,6871, Sent = 126, SentPerMin = 1034,18, WordPerSec = 912,98
Update = 24000, Epoch = 41, LR = 0,000387, AvgCost = 200,6429, Sent = 418, SentPerMin = 1010,30, WordPerSec = 927,07
Update = 24100, Epoch = 41, LR = 0,000386, AvgCost = 197,9179, Sent = 715, SentPerMin = 1025,73, WordPerSec = 938,03
Update = 24196, Epoch = 41, LR = 0,000386, AvgCost = 198,2280, Sent = 1000, SentPerMin = 1025,44, WordPerSec = 941,12
Update = 24200, Epoch = 42, LR = 0,000386, AvgCost = 176,8588, Sent = 12, SentPerMin = 1039,38, WordPerSec = 935,44
Update = 24300, Epoch = 42, LR = 0,000385, AvgCost = 206,1572, Sent = 299, SentPerMin = 1005,37, WordPerSec = 937,78
Update = 24400, Epoch = 42, LR = 0,000384, AvgCost = 199,5520, Sent = 601, SentPerMin = 1020,47, WordPerSec = 937,44
Update = 24500, Epoch = 42, LR = 0,000383, AvgCost = 198,8433, Sent = 897, SentPerMin = 1024,55, WordPerSec = 942,41
Update = 24534, Epoch = 42, LR = 0,000383, AvgCost = 197,8877, Sent = 1000, SentPerMin = 1029,47, WordPerSec = 944,81
Update = 24600, Epoch = 43, LR = 0,000383, AvgCost = 204,0813, Sent = 189, SentPerMin = 1003,48, WordPerSec = 924,37
Update = 24700, Epoch = 43, LR = 0,000382, AvgCost = 199,2837, Sent = 490, SentPerMin = 1022,44, WordPerSec = 934,70
Update = 24800, Epoch = 43, LR = 0,000381, AvgCost = 199,5381, Sent = 784, SentPerMin = 1017,10, WordPerSec = 937,38
Update = 24872, Epoch = 43, LR = 0,000380, AvgCost = 197,3426, Sent = 1000, SentPerMin = 1027,62, WordPerSec = 943,11
Update = 24900, Epoch = 44, LR = 0,000380, AvgCost = 192,4068, Sent = 87, SentPerMin = 1050,82, WordPerSec = 923,39
Update = 25000, Epoch = 44, LR = 0,000379, AvgCost = 200,9169, Sent = 375, SentPerMin = 1011,69, WordPerSec = 932,46
Update = 25100, Epoch = 44, LR = 0,000379, AvgCost = 198,7066, Sent = 668, SentPerMin = 1019,57, WordPerSec = 939,51
Update = 25200, Epoch = 44, LR = 0,000378, AvgCost = 196,0565, Sent = 974, SentPerMin = 1033,01, WordPerSec = 945,28
Update = 25210, Epoch = 44, LR = 0,000378, AvgCost = 196,9791, Sent = 1000, SentPerMin = 1030,45, WordPerSec = 945,72
Update = 25300, Epoch = 45, LR = 0,000377, AvgCost = 204,9573, Sent = 258, SentPerMin = 1002,46, WordPerSec = 934,40
Update = 25400, Epoch = 45, LR = 0,000376, AvgCost = 197,7264, Sent = 562, SentPerMin = 1023,42, WordPerSec = 938,41
Update = 25500, Epoch = 45, LR = 0,000376, AvgCost = 197,8932, Sent = 854, SentPerMin = 1022,46, WordPerSec = 941,57
Update = 25548, Epoch = 45, LR = 0,000375, AvgCost = 196,6862, Sent = 1000, SentPerMin = 1029,78, WordPerSec = 945,10
Update = 25600, Epoch = 46, LR = 0,000375, AvgCost = 198,1307, Sent = 152, SentPerMin = 842,92, WordPerSec = 758,35
Update = 25700, Epoch = 46, LR = 0,000374, AvgCost = 202,9924, Sent = 443, SentPerMin = 834,23, WordPerSec = 771,30
Update = 25800, Epoch = 46, LR = 0,000374, AvgCost = 198,2958, Sent = 744, SentPerMin = 842,83, WordPerSec = 774,09
Update = 25886, Epoch = 46, LR = 0,000373, AvgCost = 198,0156, Sent = 1000, SentPerMin = 848,03, WordPerSec = 778,29
Update = 25900, Epoch = 47, LR = 0,000373, AvgCost = 186,8782, Sent = 45, SentPerMin = 1105,63, WordPerSec = 905,39
Update = 26000, Epoch = 47, LR = 0,000372, AvgCost = 202,9696, Sent = 330, SentPerMin = 1016,33, WordPerSec = 942,11
Update = 26100, Epoch = 47, LR = 0,000371, AvgCost = 198,1613, Sent = 630, SentPerMin = 1025,50, WordPerSec = 940,45
Update = 26200, Epoch = 47, LR = 0,000371, AvgCost = 197,5403, Sent = 929, SentPerMin = 1029,96, WordPerSec = 946,27
Update = 26224, Epoch = 47, LR = 0,000371, AvgCost = 197,0692, Sent = 1000, SentPerMin = 1033,66, WordPerSec = 948,66
Update = 26300, Epoch = 48, LR = 0,000370, AvgCost = 200,8471, Sent = 219, SentPerMin = 1020,46, WordPerSec = 940,31
Update = 26400, Epoch = 48, LR = 0,000369, AvgCost = 199,0497, Sent = 519, SentPerMin = 1025,32, WordPerSec = 940,86
Update = 26500, Epoch = 48, LR = 0,000369, AvgCost = 197,7677, Sent = 814, SentPerMin = 1024,86, WordPerSec = 942,67
Update = 26562, Epoch = 48, LR = 0,000368, AvgCost = 196,5375, Sent = 1000, SentPerMin = 1029,71, WordPerSec = 945,03
Update = 26600, Epoch = 49, LR = 0,000368, AvgCost = 182,6806, Sent = 119, SentPerMin = 1022,20, WordPerSec = 869,59
Update = 26700, Epoch = 49, LR = 0,000367, AvgCost = 198,3488, Sent = 406, SentPerMin = 982,45, WordPerSec = 902,96
Update = 26800, Epoch = 49, LR = 0,000367, AvgCost = 195,8436, Sent = 702, SentPerMin = 998,39, WordPerSec = 913,06
Update = 26900, Epoch = 49, LR = 0,000366, AvgCost = 196,1094, Sent = 1000, SentPerMin = 1001,78, WordPerSec = 919,40
Update = 26900, Epoch = 49, LR = 0,000366, AvgCost = 196,1094, Sent = 1000, SentPerMin = 1001,78, WordPerSec = 919,40
Update = 27000, Epoch = 50, LR = 0,000365, AvgCost = 204,9668, Sent = 285, SentPerMin = 976,68, WordPerSec = 917,11
Update = 27100, Epoch = 50, LR = 0,000364, AvgCost = 196,7032, Sent = 592, SentPerMin = 1000,97, WordPerSec = 918,21
Update = 27200, Epoch = 50, LR = 0,000364, AvgCost = 197,4803, Sent = 882, SentPerMin = 995,88, WordPerSec = 920,02
Update = 27238, Epoch = 50, LR = 0,000364, AvgCost = 195,7721, Sent = 1000, SentPerMin = 1004,71, WordPerSec = 922,09
Update = 27300, Epoch = 51, LR = 0,000363, AvgCost = 197,6147, Sent = 178, SentPerMin = 977,96, WordPerSec = 902,41
Update = 27400, Epoch = 51, LR = 0,000362, AvgCost = 198,6741, Sent = 475, SentPerMin = 988,36, WordPerSec = 910,40
Update = 27500, Epoch = 51, LR = 0,000362, AvgCost = 197,7047, Sent = 771, SentPerMin = 989,05, WordPerSec = 914,18
Update = 27576, Epoch = 51, LR = 0,000361, AvgCost = 195,3875, Sent = 1000, SentPerMin = 1000,96, WordPerSec = 918,65
Update = 27600, Epoch = 52, LR = 0,000361, AvgCost = 189,6840, Sent = 76, SentPerMin = 1019,66, WordPerSec = 899,35
Update = 27700, Epoch = 52, LR = 0,000361, AvgCost = 198,9727, Sent = 363, SentPerMin = 986,78, WordPerSec = 909,94
Update = 27800, Epoch = 52, LR = 0,000360, AvgCost = 196,5948, Sent = 656, SentPerMin = 992,13, WordPerSec = 913,49
Update = 27900, Epoch = 52, LR = 0,000359, AvgCost = 194,7445, Sent = 961, SentPerMin = 1001,08, WordPerSec = 917,55
Update = 27914, Epoch = 52, LR = 0,000359, AvgCost = 195,1083, Sent = 1000, SentPerMin = 1001,03, WordPerSec = 918,71
Update = 28000, Epoch = 53, LR = 0,000359, AvgCost = 198,8732, Sent = 248, SentPerMin = 981,05, WordPerSec = 906,28
Update = 28100, Epoch = 53, LR = 0,000358, AvgCost = 195,9330, Sent = 549, SentPerMin = 992,96, WordPerSec = 909,79
Update = 28200, Epoch = 53, LR = 0,000357, AvgCost = 195,9752, Sent = 841, SentPerMin = 991,75, WordPerSec = 913,25
Update = 28252, Epoch = 53, LR = 0,000357, AvgCost = 194,7454, Sent = 1000, SentPerMin = 999,22, WordPerSec = 917,05
Update = 28300, Epoch = 54, LR = 0,000357, AvgCost = 193,2790, Sent = 140, SentPerMin = 988,37, WordPerSec = 887,18
Update = 28400, Epoch = 54, LR = 0,000356, AvgCost = 199,9784, Sent = 431, SentPerMin = 975,89, WordPerSec = 904,60
Update = 28500, Epoch = 54, LR = 0,000355, AvgCost = 194,0388, Sent = 732, SentPerMin = 995,63, WordPerSec = 910,25
Update = 28590, Epoch = 54, LR = 0,000355, AvgCost = 194,5143, Sent = 1000, SentPerMin = 997,36, WordPerSec = 915,34
Update = 28600, Epoch = 55, LR = 0,000355, AvgCost = 159,7876, Sent = 33, SentPerMin = 1102,41, WordPerSec = 851,86
Update = 28700, Epoch = 55, LR = 0,000354, AvgCost = 200,3348, Sent = 317, SentPerMin = 971,34, WordPerSec = 904,54
Update = 28800, Epoch = 55, LR = 0,000354, AvgCost = 195,3903, Sent = 618, SentPerMin = 991,69, WordPerSec = 910,15
Update = 28900, Epoch = 55, LR = 0,000353, AvgCost = 195,2148, Sent = 914, SentPerMin = 994,44, WordPerSec = 916,38
Update = 28928, Epoch = 55, LR = 0,000353, AvgCost = 194,1098, Sent = 1000, SentPerMin = 999,85, WordPerSec = 917,63
Update = 29000, Epoch = 56, LR = 0,000352, AvgCost = 203,0858, Sent = 205, SentPerMin = 791,45, WordPerSec = 738,94
Update = 29100, Epoch = 56, LR = 0,000352, AvgCost = 197,3858, Sent = 506, SentPerMin = 809,06, WordPerSec = 742,09
Update = 29200, Epoch = 56, LR = 0,000351, AvgCost = 197,3244, Sent = 802, SentPerMin = 812,36, WordPerSec = 747,88
Update = 29266, Epoch = 56, LR = 0,000351, AvgCost = 195,7296, Sent = 1000, SentPerMin = 818,32, WordPerSec = 751,03
Update = 29300, Epoch = 57, LR = 0,000351, AvgCost = 185,8205, Sent = 106, SentPerMin = 1024,24, WordPerSec = 882,19
Update = 29400, Epoch = 57, LR = 0,000350, AvgCost = 198,0610, Sent = 393, SentPerMin = 978,45, WordPerSec = 901,77
Update = 29500, Epoch = 57, LR = 0,000349, AvgCost = 195,1041, Sent = 689, SentPerMin = 995,67, WordPerSec = 912,70
Update = 29600, Epoch = 57, LR = 0,000349, AvgCost = 194,9853, Sent = 989, SentPerMin = 998,48, WordPerSec = 916,62
Update = 29604, Epoch = 57, LR = 0,000349, AvgCost = 194,8304, Sent = 1000, SentPerMin = 998,55, WordPerSec = 916,44
Update = 29700, Epoch = 58, LR = 0,000348, AvgCost = 203,9556, Sent = 275, SentPerMin = 972,58, WordPerSec = 912,99
Update = 29800, Epoch = 58, LR = 0,000348, AvgCost = 195,4959, Sent = 581, SentPerMin = 992,88, WordPerSec = 910,83
Update = 29900, Epoch = 58, LR = 0,000347, AvgCost = 195,6250, Sent = 870, SentPerMin = 993,15, WordPerSec = 916,95
Update = 29942, Epoch = 58, LR = 0,000347, AvgCost = 194,2556, Sent = 1000, SentPerMin = 1000,38, WordPerSec = 918,12
Update = 30000, Epoch = 59, LR = 0,000346, AvgCost = 197,5074, Sent = 166, SentPerMin = 961,17, WordPerSec = 885,71
Update = 30100, Epoch = 59, LR = 0,000346, AvgCost = 197,7074, Sent = 463, SentPerMin = 731,38, WordPerSec = 675,06
Update = 30200, Epoch = 59, LR = 0,000345, AvgCost = 195,5005, Sent = 761, SentPerMin = 815,79, WordPerSec = 751,22
Update = 30280, Epoch = 59, LR = 0,000345, AvgCost = 193,8724, Sent = 1000, SentPerMin = 858,51, WordPerSec = 787,91
Update = 30300, Epoch = 60, LR = 0,000345, AvgCost = 188,5431, Sent = 64, SentPerMin = 1032,52, WordPerSec = 900,23
Update = 30400, Epoch = 60, LR = 0,000344, AvgCost = 197,4508, Sent = 352, SentPerMin = 980,59, WordPerSec = 901,75
Update = 30500, Epoch = 60, LR = 0,000344, AvgCost = 194,8499, Sent = 645, SentPerMin = 987,07, WordPerSec = 908,56
Update = 30600, Epoch = 60, LR = 0,000343, AvgCost = 194,0277, Sent = 948, SentPerMin = 995,30, WordPerSec = 914,98
Update = 30618, Epoch = 60, LR = 0,000343, AvgCost = 193,5563, Sent = 1000, SentPerMin = 997,62, WordPerSec = 915,58
Update = 30700, Epoch = 61, LR = 0,000342, AvgCost = 195,3176, Sent = 239, SentPerMin = 989,88, WordPerSec = 904,00
Update = 30800, Epoch = 61, LR = 0,000342, AvgCost = 195,4012, Sent = 536, SentPerMin = 987,28, WordPerSec = 907,50
Update = 30900, Epoch = 61, LR = 0,000341, AvgCost = 193,8019, Sent = 831, SentPerMin = 991,69, WordPerSec = 910,60
Update = 30956, Epoch = 61, LR = 0,000341, AvgCost = 193,1425, Sent = 1000, SentPerMin = 997,09, WordPerSec = 915,10
Update = 31000, Epoch = 62, LR = 0,000341, AvgCost = 190,9224, Sent = 131, SentPerMin = 1008,75, WordPerSec = 897,74
Update = 31100, Epoch = 62, LR = 0,000340, AvgCost = 195,1442, Sent = 423, SentPerMin = 989,43, WordPerSec = 909,27
Update = 31200, Epoch = 62, LR = 0,000340, AvgCost = 192,4545, Sent = 720, SentPerMin = 1001,05, WordPerSec = 916,12
Update = 31294, Epoch = 62, LR = 0,000339, AvgCost = 192,7644, Sent = 1000, SentPerMin = 1000,71, WordPerSec = 918,42
Update = 31300, Epoch = 63, LR = 0,000339, AvgCost = 190,9656, Sent = 16, SentPerMin = 907,35, WordPerSec = 885,62
Update = 31400, Epoch = 63, LR = 0,000339, AvgCost = 199,7206, Sent = 304, SentPerMin = 971,27, WordPerSec = 907,16
Update = 31500, Epoch = 63, LR = 0,000338, AvgCost = 193,7606, Sent = 606, SentPerMin = 982,74, WordPerSec = 903,19
Update = 31600, Epoch = 63, LR = 0,000338, AvgCost = 193,5418, Sent = 902, SentPerMin = 988,99, WordPerSec = 910,85
Update = 31632, Epoch = 63, LR = 0,000337, AvgCost = 192,3198, Sent = 1000, SentPerMin = 994,50, WordPerSec = 912,72
Update = 31700, Epoch = 64, LR = 0,000337, AvgCost = 197,3007, Sent = 195, SentPerMin = 981,19, WordPerSec = 901,86
Update = 31800, Epoch = 64, LR = 0,000336, AvgCost = 193,6189, Sent = 495, SentPerMin = 995,24, WordPerSec = 909,56
Update = 31900, Epoch = 64, LR = 0,000336, AvgCost = 194,1649, Sent = 789, SentPerMin = 990,25, WordPerSec = 913,08
Update = 31970, Epoch = 64, LR = 0,000336, AvgCost = 192,0901, Sent = 1000, SentPerMin = 998,99, WordPerSec = 916,84
Update = 32000, Epoch = 65, LR = 0,000335, AvgCost = 189,5182, Sent = 92, SentPerMin = 1008,00, WordPerSec = 894,60
Update = 32100, Epoch = 65, LR = 0,000335, AvgCost = 195,0577, Sent = 381, SentPerMin = 987,99, WordPerSec = 909,16
Update = 32200, Epoch = 65, LR = 0,000334, AvgCost = 192,7720, Sent = 676, SentPerMin = 995,37, WordPerSec = 913,82
Update = 32300, Epoch = 65, LR = 0,000334, AvgCost = 190,8091, Sent = 979, SentPerMin = 997,25, WordPerSec = 913,14
Update = 32308, Epoch = 65, LR = 0,000334, AvgCost = 191,7324, Sent = 1000, SentPerMin = 995,11, WordPerSec = 913,28
Update = 32400, Epoch = 66, LR = 0,000333, AvgCost = 200,7833, Sent = 265, SentPerMin = 806,35, WordPerSec = 748,34
Update = 32500, Epoch = 66, LR = 0,000333, AvgCost = 195,6634, Sent = 566, SentPerMin = 815,35, WordPerSec = 750,43
Update = 32600, Epoch = 66, LR = 0,000332, AvgCost = 194,8274, Sent = 859, SentPerMin = 813,64, WordPerSec = 749,77
Update = 32646, Epoch = 66, LR = 0,000332, AvgCost = 193,5357, Sent = 1000, SentPerMin = 819,21, WordPerSec = 751,84
Update = 32700, Epoch = 67, LR = 0,000332, AvgCost = 195,4047, Sent = 157, SentPerMin = 981,15, WordPerSec = 892,52
Update = 32800, Epoch = 67, LR = 0,000331, AvgCost = 197,3501, Sent = 450, SentPerMin = 981,60, WordPerSec = 905,15
Update = 32900, Epoch = 67, LR = 0,000331, AvgCost = 192,8594, Sent = 752, SentPerMin = 989,79, WordPerSec = 907,46
Update = 32984, Epoch = 67, LR = 0,000330, AvgCost = 192,8595, Sent = 1000, SentPerMin = 994,26, WordPerSec = 912,50
Update = 33000, Epoch = 68, LR = 0,000330, AvgCost = 185,8780, Sent = 51, SentPerMin = 1024,93, WordPerSec = 883,59
Update = 33100, Epoch = 68, LR = 0,000330, AvgCost = 198,1749, Sent = 336, SentPerMin = 973,35, WordPerSec = 901,80
Update = 33200, Epoch = 68, LR = 0,000329, AvgCost = 192,8548, Sent = 636, SentPerMin = 985,99, WordPerSec = 903,39
Update = 33300, Epoch = 68, LR = 0,000329, AvgCost = 192,7471, Sent = 935, SentPerMin = 988,32, WordPerSec = 908,52
Update = 33322, Epoch = 68, LR = 0,000329, AvgCost = 192,2999, Sent = 1000, SentPerMin = 991,28, WordPerSec = 909,76
Update = 33400, Epoch = 69, LR = 0,000328, AvgCost = 195,2722, Sent = 225, SentPerMin = 980,03, WordPerSec = 900,90
Update = 33500, Epoch = 69, LR = 0,000328, AvgCost = 194,5821, Sent = 524, SentPerMin = 983,58, WordPerSec = 905,03
Update = 33600, Epoch = 69, LR = 0,000327, AvgCost = 193,0491, Sent = 819, SentPerMin = 988,03, WordPerSec = 908,79
Update = 33660, Epoch = 69, LR = 0,000327, AvgCost = 191,9609, Sent = 1000, SentPerMin = 995,35, WordPerSec = 913,50
Update = 33700, Epoch = 70, LR = 0,000327, AvgCost = 187,6552, Sent = 122, SentPerMin = 1020,51, WordPerSec = 890,71
Update = 33800, Epoch = 70, LR = 0,000326, AvgCost = 194,7345, Sent = 411, SentPerMin = 978,18, WordPerSec = 900,91
Update = 33900, Epoch = 70, LR = 0,000326, AvgCost = 191,2177, Sent = 709, SentPerMin = 978,63, WordPerSec = 893,92
Update = 33998, Epoch = 70, LR = 0,000325, AvgCost = 191,6581, Sent = 1000, SentPerMin = 983,13, WordPerSec = 902,28
Update = 34000, Epoch = 71, LR = 0,000325, AvgCost = 167,5963, Sent = 6, SentPerMin = 989,15, WordPerSec = 898,48
Update = 34100, Epoch = 71, LR = 0,000325, AvgCost = 199,6534, Sent = 292, SentPerMin = 970,13, WordPerSec = 907,67
Update = 34200, Epoch = 71, LR = 0,000324, AvgCost = 193,4580, Sent = 595, SentPerMin = 989,04, WordPerSec = 910,31
Update = 34300, Epoch = 71, LR = 0,000324, AvgCost = 192,7755, Sent = 890, SentPerMin = 992,42, WordPerSec = 914,83
Update = 34336, Epoch = 71, LR = 0,000324, AvgCost = 191,4240, Sent = 1000, SentPerMin = 998,69, WordPerSec = 916,56
Update = 34400, Epoch = 72, LR = 0,000323, AvgCost = 196,8425, Sent = 184, SentPerMin = 948,24, WordPerSec = 871,19
Update = 34500, Epoch = 72, LR = 0,000323, AvgCost = 193,7559, Sent = 482, SentPerMin = 939,09, WordPerSec = 863,36
Update = 34600, Epoch = 72, LR = 0,000323, AvgCost = 193,8575, Sent = 775, SentPerMin = 934,92, WordPerSec = 866,22
Update = 34674, Epoch = 72, LR = 0,000322, AvgCost = 190,9942, Sent = 1000, SentPerMin = 948,83, WordPerSec = 870,81
Update = 34700, Epoch = 73, LR = 0,000322, AvgCost = 185,7840, Sent = 81, SentPerMin = 915,17, WordPerSec = 812,36
Update = 34800, Epoch = 73, LR = 0,000322, AvgCost = 195,5692, Sent = 368, SentPerMin = 880,77, WordPerSec = 815,11
Update = 34900, Epoch = 73, LR = 0,000321, AvgCost = 192,0146, Sent = 663, SentPerMin = 899,83, WordPerSec = 826,75
Update = 35000, Epoch = 73, LR = 0,000321, AvgCost = 190,0078, Sent = 969, SentPerMin = 935,96, WordPerSec = 855,87
Update = 35012, Epoch = 73, LR = 0,000321, AvgCost = 190,8008, Sent = 1000, SentPerMin = 934,67, WordPerSec = 857,81
Update = 35100, Epoch = 74, LR = 0,000320, AvgCost = 199,5136, Sent = 251, SentPerMin = 871,74, WordPerSec = 814,38
Update = 35200, Epoch = 74, LR = 0,000320, AvgCost = 191,5125, Sent = 556, SentPerMin = 829,36, WordPerSec = 759,13
Update = 35300, Epoch = 74, LR = 0,000319, AvgCost = 191,8359, Sent = 847, SentPerMin = 807,84, WordPerSec = 744,59
Update = 35350, Epoch = 74, LR = 0,000319, AvgCost = 190,5379, Sent = 1000, SentPerMin = 808,75, WordPerSec = 742,24
Update = 35400, Epoch = 75, LR = 0,000319, AvgCost = 192,5266, Sent = 145, SentPerMin = 768,72, WordPerSec = 699,18
Update = 35500, Epoch = 75, LR = 0,000318, AvgCost = 195,6424, Sent = 437, SentPerMin = 764,26, WordPerSec = 708,24
Update = 35600, Epoch = 75, LR = 0,000318, AvgCost = 190,1981, Sent = 739, SentPerMin = 777,38, WordPerSec = 712,61
Update = 35688, Epoch = 75, LR = 0,000318, AvgCost = 190,2689, Sent = 1000, SentPerMin = 781,65, WordPerSec = 717,37
Update = 35700, Epoch = 76, LR = 0,000318, AvgCost = 155,3412, Sent = 41, SentPerMin = 731,27, WordPerSec = 563,02
Update = 35800, Epoch = 76, LR = 0,000317, AvgCost = 198,0412, Sent = 324, SentPerMin = 634,15, WordPerSec = 589,03
Update = 35900, Epoch = 76, LR = 0,000317, AvgCost = 192,8699, Sent = 624, SentPerMin = 643,24, WordPerSec = 589,69
Update = 36000, Epoch = 76, LR = 0,000316, AvgCost = 192,5278, Sent = 921, SentPerMin = 645,93, WordPerSec = 594,29
Update = 36026, Epoch = 76, LR = 0,000316, AvgCost = 191,7739, Sent = 1000, SentPerMin = 647,87, WordPerSec = 594,59
Update = 36100, Epoch = 77, LR = 0,000316, AvgCost = 196,3179, Sent = 213, SentPerMin = 767,20, WordPerSec = 707,77
Update = 36200, Epoch = 77, LR = 0,000315, AvgCost = 194,4702, Sent = 511, SentPerMin = 773,82, WordPerSec = 713,55
Update = 36300, Epoch = 77, LR = 0,000315, AvgCost = 192,8018, Sent = 808, SentPerMin = 780,63, WordPerSec = 719,05
Update = 36364, Epoch = 77, LR = 0,000315, AvgCost = 191,1034, Sent = 1000, SentPerMin = 785,81, WordPerSec = 721,19
Update = 36400, Epoch = 78, LR = 0,000314, AvgCost = 180,3433, Sent = 113, SentPerMin = 823,94, WordPerSec = 705,82
Update = 36500, Epoch = 78, LR = 0,000314, AvgCost = 193,3504, Sent = 399, SentPerMin = 773,91, WordPerSec = 712,36
Update = 36600, Epoch = 78, LR = 0,000314, AvgCost = 191,1108, Sent = 696, SentPerMin = 783,02, WordPerSec = 717,42
Update = 36700, Epoch = 78, LR = 0,000313, AvgCost = 190,6456, Sent = 995, SentPerMin = 784,87, WordPerSec = 719,83
Update = 36702, Epoch = 78, LR = 0,000313, AvgCost = 190,5962, Sent = 1000, SentPerMin = 784,39, WordPerSec = 719,88
Update = 36800, Epoch = 79, LR = 0,000313, AvgCost = 199,7620, Sent = 280, SentPerMin = 759,53, WordPerSec = 712,69
Update = 36900, Epoch = 79, LR = 0,000312, AvgCost = 191,0375, Sent = 588, SentPerMin = 812,16, WordPerSec = 742,92
Update = 37000, Epoch = 79, LR = 0,000312, AvgCost = 191,5388, Sent = 877, SentPerMin = 839,46, WordPerSec = 774,20
Update = 37040, Epoch = 79, LR = 0,000312, AvgCost = 190,3209, Sent = 1000, SentPerMin = 853,05, WordPerSec = 782,90
Update = 37100, Epoch = 80, LR = 0,000312, AvgCost = 192,0025, Sent = 173, SentPerMin = 893,40, WordPerSec = 821,96
Update = 37200, Epoch = 80, LR = 0,000311, AvgCost = 193,6223, Sent = 469, SentPerMin = 902,63, WordPerSec = 832,61
Update = 37300, Epoch = 80, LR = 0,000311, AvgCost = 192,4034, Sent = 766, SentPerMin = 899,22, WordPerSec = 829,74
Update = 37378, Epoch = 80, LR = 0,000310, AvgCost = 190,1223, Sent = 1000, SentPerMin = 905,93, WordPerSec = 831,43
Update = 37400, Epoch = 81, LR = 0,000310, AvgCost = 182,1746, Sent = 71, SentPerMin = 929,03, WordPerSec = 804,07
Update = 37500, Epoch = 81, LR = 0,000310, AvgCost = 193,1797, Sent = 358, SentPerMin = 893,27, WordPerSec = 820,54
Update = 37600, Epoch = 81, LR = 0,000309, AvgCost = 190,9763, Sent = 652, SentPerMin = 903,12, WordPerSec = 829,31
Update = 37700, Epoch = 81, LR = 0,000309, AvgCost = 189,8183, Sent = 955, SentPerMin = 909,49, WordPerSec = 834,39
Update = 37716, Epoch = 81, LR = 0,000309, AvgCost = 189,8156, Sent = 1000, SentPerMin = 909,47, WordPerSec = 834,68
Update = 37800, Epoch = 82, LR = 0,000309, AvgCost = 191,2621, Sent = 244, SentPerMin = 902,17, WordPerSec = 824,22
Update = 37900, Epoch = 82, LR = 0,000308, AvgCost = 191,2436, Sent = 542, SentPerMin = 898,11, WordPerSec = 824,87
Update = 38000, Epoch = 82, LR = 0,000308, AvgCost = 190,7497, Sent = 835, SentPerMin = 900,49, WordPerSec = 828,47
Update = 38054, Epoch = 82, LR = 0,000308, AvgCost = 189,5600, Sent = 1000, SentPerMin = 908,20, WordPerSec = 833,52
Update = 38100, Epoch = 83, LR = 0,000307, AvgCost = 185,8881, Sent = 136, SentPerMin = 924,07, WordPerSec = 822,26
Update = 38200, Epoch = 83, LR = 0,000307, AvgCost = 191,4328, Sent = 428, SentPerMin = 900,15, WordPerSec = 827,94
Update = 38300, Epoch = 83, LR = 0,000307, AvgCost = 189,4087, Sent = 725, SentPerMin = 906,70, WordPerSec = 831,14
Update = 38392, Epoch = 83, LR = 0,000306, AvgCost = 189,1535, Sent = 1000, SentPerMin = 907,89, WordPerSec = 833,23
Update = 38400, Epoch = 84, LR = 0,000306, AvgCost = 171,6225, Sent = 23, SentPerMin = 864,76, WordPerSec = 753,84
Update = 38500, Epoch = 84, LR = 0,000306, AvgCost = 194,9248, Sent = 311, SentPerMin = 889,03, WordPerSec = 826,57
Update = 38600, Epoch = 84, LR = 0,000305, AvgCost = 190,7089, Sent = 611, SentPerMin = 896,95, WordPerSec = 826,12
Update = 38700, Epoch = 84, LR = 0,000305, AvgCost = 190,2505, Sent = 906, SentPerMin = 898,97, WordPerSec = 829,30
Update = 38730, Epoch = 84, LR = 0,000305, AvgCost = 188,8884, Sent = 1000, SentPerMin = 905,34, WordPerSec = 830,89
Update = 38800, Epoch = 85, LR = 0,000305, AvgCost = 193,3172, Sent = 200, SentPerMin = 729,57, WordPerSec = 669,50
Update = 38900, Epoch = 85, LR = 0,000304, AvgCost = 191,7193, Sent = 501, SentPerMin = 738,71, WordPerSec = 676,75
Update = 39000, Epoch = 85, LR = 0,000304, AvgCost = 192,4434, Sent = 795, SentPerMin = 738,29, WordPerSec = 680,26
Update = 39068, Epoch = 85, LR = 0,000304, AvgCost = 190,4101, Sent = 1000, SentPerMin = 745,85, WordPerSec = 684,51
Starting inference...
Inference results:
) 的在了••• () ) 的了在了

了和了 和
) 的了在了"了。
。


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
                EncoderLayerDepth = 2,
                DecoderLayerDepth = 2,
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
                MaxEpochNum = 86, // 73, // 66, // 50, // 40, // 26, // 13, // 3, 
                SharedEmbeddings = false,
                ModelFilePath = "seq2seq_test86.model",
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
