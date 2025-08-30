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
// and put data files in exe folder
// The training files contain only 1000 sentence pairs. 
namespace ConsoleApplication4May2025
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

            // https://huggingface.co/spaces/zhongkaifu/mt_enu_chs
            /*
            您好,谁是参加展览会的意大利编辑?
            这是1980年代的一个转折点。
            我喜欢早鸟和全包优惠等折扣。
            红地毯挤在机场机库。
            许多人在军事线上等待。
            EOS
            */

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
             Update = 10100, Epoch = 0, LR = 0,000597, AvgCost = 266,5767, Sent = 285, SentPerMin = 100,15, WordPerSec = 94,05
Update = 10200, Epoch = 0, LR = 0,000594, AvgCost = 248,2335, Sent = 592, SentPerMin = 102,22, WordPerSec = 93,77
Update = 10300, Epoch = 0, LR = 0,000591, AvgCost = 245,5753, Sent = 882, SentPerMin = 102,83, WordPerSec = 95,00
Update = 10338, Epoch = 0, LR = 0,000590, AvgCost = 242,3933, Sent = 1000, SentPerMin = 103,74, WordPerSec = 95,21
Update = 10400, Epoch = 1, LR = 0,000588, AvgCost = 232,9623, Sent = 178, SentPerMin = 142,93, WordPerSec = 131,89
Update = 10500, Epoch = 1, LR = 0,000586, AvgCost = 233,8012, Sent = 475, SentPerMin = 143,00, WordPerSec = 131,72
Update = 10600, Epoch = 1, LR = 0,000583, AvgCost = 232,2756, Sent = 771, SentPerMin = 143,18, WordPerSec = 132,34
Update = 10676, Epoch = 1, LR = 0,000581, AvgCost = 229,1094, Sent = 1000, SentPerMin = 145,45, WordPerSec = 133,49
Update = 10700, Epoch = 2, LR = 0,000580, AvgCost = 220,7516, Sent = 76, SentPerMin = 104,08, WordPerSec = 91,80
Update = 10800, Epoch = 2, LR = 0,000577, AvgCost = 231,1393, Sent = 363, SentPerMin = 102,22, WordPerSec = 94,26
Update = 10900, Epoch = 2, LR = 0,000575, AvgCost = 227,7077, Sent = 656, SentPerMin = 102,99, WordPerSec = 94,83
Update = 11000, Epoch = 2, LR = 0,000572, AvgCost = 225,2301, Sent = 961, SentPerMin = 103,59, WordPerSec = 94,94
Update = 11014, Epoch = 2, LR = 0,000572, AvgCost = 225,6480, Sent = 1000, SentPerMin = 103,72, WordPerSec = 95,19
Starting inference...
Inference results:
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
             */


            /*
             Update = 10100, Epoch = 0, LR = 0,000597, AvgCost = 266,6459, Sent = 285, SentPerMin = 99,93, WordPerSec = 93,83
Update = 10200, Epoch = 0, LR = 0,000594, AvgCost = 248,0130, Sent = 592, SentPerMin = 101,21, WordPerSec = 92,84
Update = 10300, Epoch = 0, LR = 0,000591, AvgCost = 245,2522, Sent = 882, SentPerMin = 101,81, WordPerSec = 94,05
Update = 10338, Epoch = 0, LR = 0,000590, AvgCost = 242,1006, Sent = 1000, SentPerMin = 102,70, WordPerSec = 94,26
Update = 10400, Epoch = 1, LR = 0,000588, AvgCost = 233,0658, Sent = 178, SentPerMin = 142,17, WordPerSec = 131,19
Update = 10500, Epoch = 1, LR = 0,000586, AvgCost = 233,8489, Sent = 475, SentPerMin = 142,10, WordPerSec = 130,89
Update = 10600, Epoch = 1, LR = 0,000583, AvgCost = 232,3411, Sent = 771, SentPerMin = 142,13, WordPerSec = 131,37
Update = 10676, Epoch = 1, LR = 0,000581, AvgCost = 229,1570, Sent = 1000, SentPerMin = 144,33, WordPerSec = 132,46
Update = 10700, Epoch = 2, LR = 0,000580, AvgCost = 219,7913, Sent = 76, SentPerMin = 103,28, WordPerSec = 91,10
Update = 10800, Epoch = 2, LR = 0,000577, AvgCost = 230,3476, Sent = 363, SentPerMin = 102,26, WordPerSec = 94,30
Update = 10900, Epoch = 2, LR = 0,000575, AvgCost = 227,1786, Sent = 656, SentPerMin = 102,88, WordPerSec = 94,72
Update = 11000, Epoch = 2, LR = 0,000572, AvgCost = 224,6107, Sent = 961, SentPerMin = 103,61, WordPerSec = 94,96
Update = 11014, Epoch = 2, LR = 0,000572, AvgCost = 224,9827, Sent = 1000, SentPerMin = 103,73, WordPerSec = 95,20
Update = 11100, Epoch = 3, LR = 0,000569, AvgCost = 228,0399, Sent = 248, SentPerMin = 143,48, WordPerSec = 132,55
Update = 11200, Epoch = 3, LR = 0,000567, AvgCost = 224,1225, Sent = 549, SentPerMin = 142,87, WordPerSec = 130,91
Update = 11300, Epoch = 3, LR = 0,000564, AvgCost = 224,1251, Sent = 841, SentPerMin = 143,23, WordPerSec = 131,89
Update = 11352, Epoch = 3, LR = 0,000563, AvgCost = 222,7356, Sent = 1000, SentPerMin = 144,27, WordPerSec = 132,41
Update = 11400, Epoch = 4, LR = 0,000562, AvgCost = 220,2004, Sent = 140, SentPerMin = 105,41, WordPerSec = 94,62
Update = 11500, Epoch = 4, LR = 0,000560, AvgCost = 227,6601, Sent = 431, SentPerMin = 102,02, WordPerSec = 94,57
Update = 11600, Epoch = 4, LR = 0,000557, AvgCost = 220,8914, Sent = 732, SentPerMin = 103,75, WordPerSec = 94,86
Update = 11690, Epoch = 4, LR = 0,000555, AvgCost = 221,2590, Sent = 1000, SentPerMin = 103,68, WordPerSec = 95,16
Update = 11700, Epoch = 5, LR = 0,000555, AvgCost = 183,3972, Sent = 33, SentPerMin = 159,02, WordPerSec = 122,88
Update = 11800, Epoch = 5, LR = 0,000552, AvgCost = 228,0588, Sent = 317, SentPerMin = 142,11, WordPerSec = 132,34
Update = 11900, Epoch = 5, LR = 0,000550, AvgCost = 221,7577, Sent = 618, SentPerMin = 143,29, WordPerSec = 131,51
Update = 12000, Epoch = 5, LR = 0,000548, AvgCost = 221,3119, Sent = 914, SentPerMin = 143,74, WordPerSec = 132,46
Update = 12028, Epoch = 5, LR = 0,000547, AvgCost = 220,1514, Sent = 1000, SentPerMin = 144,63, WordPerSec = 132,74
Update = 12100, Epoch = 6, LR = 0,000545, AvgCost = 228,4766, Sent = 205, SentPerMin = 102,14, WordPerSec = 95,36
Update = 12200, Epoch = 6, LR = 0,000543, AvgCost = 221,3772, Sent = 506, SentPerMin = 102,57, WordPerSec = 94,08
Update = 12300, Epoch = 6, LR = 0,000541, AvgCost = 221,5959, Sent = 802, SentPerMin = 102,94, WordPerSec = 94,77
Update = 12366, Epoch = 6, LR = 0,000540, AvgCost = 219,5890, Sent = 1000, SentPerMin = 103,76, WordPerSec = 95,23
Update = 12400, Epoch = 7, LR = 0,000539, AvgCost = 210,2885, Sent = 106, SentPerMin = 146,58, WordPerSec = 126,25
Update = 12500, Epoch = 7, LR = 0,000537, AvgCost = 222,5211, Sent = 393, SentPerMin = 141,02, WordPerSec = 129,97
Update = 12600, Epoch = 7, LR = 0,000535, AvgCost = 219,3412, Sent = 689, SentPerMin = 144,18, WordPerSec = 132,16
Update = 12700, Epoch = 7, LR = 0,000532, AvgCost = 218,8679, Sent = 989, SentPerMin = 144,15, WordPerSec = 132,33
Update = 12704, Epoch = 7, LR = 0,000532, AvgCost = 218,7203, Sent = 1000, SentPerMin = 144,26, WordPerSec = 132,39
Update = 12800, Epoch = 8, LR = 0,000530, AvgCost = 229,6757, Sent = 275, SentPerMin = 101,73, WordPerSec = 95,50
Update = 12900, Epoch = 8, LR = 0,000528, AvgCost = 219,4538, Sent = 581, SentPerMin = 102,44, WordPerSec = 93,97
Update = 13000, Epoch = 8, LR = 0,000526, AvgCost = 219,7825, Sent = 870, SentPerMin = 103,02, WordPerSec = 95,12
Update = 13042, Epoch = 8, LR = 0,000525, AvgCost = 218,2326, Sent = 1000, SentPerMin = 103,66, WordPerSec = 95,13
Update = 13100, Epoch = 9, LR = 0,000524, AvgCost = 221,6656, Sent = 166, SentPerMin = 141,95, WordPerSec = 130,80
Update = 13200, Epoch = 9, LR = 0,000522, AvgCost = 221,3258, Sent = 463, SentPerMin = 141,32, WordPerSec = 130,44
Update = 13300, Epoch = 9, LR = 0,000520, AvgCost = 219,1346, Sent = 761, SentPerMin = 142,10, WordPerSec = 130,85
Update = 13380, Epoch = 9, LR = 0,000519, AvgCost = 217,1709, Sent = 1000, SentPerMin = 144,14, WordPerSec = 132,29
Update = 13400, Epoch = 10, LR = 0,000518, AvgCost = 212,4734, Sent = 64, SentPerMin = 106,44, WordPerSec = 92,81
Update = 13500, Epoch = 10, LR = 0,000516, AvgCost = 220,4470, Sent = 352, SentPerMin = 102,31, WordPerSec = 94,08
Update = 13600, Epoch = 10, LR = 0,000514, AvgCost = 217,5672, Sent = 645, SentPerMin = 102,77, WordPerSec = 94,60
Update = 13700, Epoch = 10, LR = 0,000513, AvgCost = 216,8434, Sent = 948, SentPerMin = 103,38, WordPerSec = 95,04
Update = 13718, Epoch = 10, LR = 0,000512, AvgCost = 216,4354, Sent = 1000, SentPerMin = 103,75, WordPerSec = 95,21
Update = 13800, Epoch = 11, LR = 0,000511, AvgCost = 217,2590, Sent = 239, SentPerMin = 143,97, WordPerSec = 131,48
Update = 13900, Epoch = 11, LR = 0,000509, AvgCost = 217,1500, Sent = 536, SentPerMin = 140,98, WordPerSec = 129,59
Update = 14000, Epoch = 11, LR = 0,000507, AvgCost = 215,7541, Sent = 831, SentPerMin = 141,91, WordPerSec = 130,30
Update = 14056, Epoch = 11, LR = 0,000506, AvgCost = 215,0929, Sent = 1000, SentPerMin = 142,67, WordPerSec = 130,93
Update = 14100, Epoch = 12, LR = 0,000505, AvgCost = 213,2146, Sent = 131, SentPerMin = 103,87, WordPerSec = 92,44
Update = 14200, Epoch = 12, LR = 0,000504, AvgCost = 216,8801, Sent = 423, SentPerMin = 101,81, WordPerSec = 93,56
Update = 14300, Epoch = 12, LR = 0,000502, AvgCost = 214,2179, Sent = 720, SentPerMin = 103,30, WordPerSec = 94,53
Update = 14394, Epoch = 12, LR = 0,000500, AvgCost = 214,6052, Sent = 1000, SentPerMin = 103,42, WordPerSec = 94,92
Starting inference...
Inference results:
,的的      ) ) 的 的    ) ) 的 的   ) ) 的的的    ) ) 的的的    ) ) 的的的
,的的的   ) 的。
,的的    ) ) 的     ) ) 的 。
,的的      ) 的     ) ) 的的的    ) ) 的的。
,的的   ) ) 的。



             */


            /*
             Update = 10100, Epoch = 0, LR = 0,000597, AvgCost = 265,5628, Sent = 285, SentPerMin = 629,93, WordPerSec = 591,51
Update = 10200, Epoch = 0, LR = 0,000594, AvgCost = 247,6976, Sent = 592, SentPerMin = 666,62, WordPerSec = 611,50
Update = 10300, Epoch = 0, LR = 0,000591, AvgCost = 244,9683, Sent = 882, SentPerMin = 670,64, WordPerSec = 619,56
Update = 10338, Epoch = 0, LR = 0,000590, AvgCost = 241,8419, Sent = 1000, SentPerMin = 677,14, WordPerSec = 621,46
Update = 10400, Epoch = 1, LR = 0,000588, AvgCost = 233,0174, Sent = 178, SentPerMin = 950,86, WordPerSec = 877,41
Update = 10500, Epoch = 1, LR = 0,000586, AvgCost = 233,7196, Sent = 475, SentPerMin = 967,22, WordPerSec = 890,93
Update = 10600, Epoch = 1, LR = 0,000583, AvgCost = 232,1876, Sent = 771, SentPerMin = 971,24, WordPerSec = 897,71
Update = 10676, Epoch = 1, LR = 0,000581, AvgCost = 229,0187, Sent = 1000, SentPerMin = 983,19, WordPerSec = 902,34
Update = 10700, Epoch = 2, LR = 0,000580, AvgCost = 221,5040, Sent = 76, SentPerMin = 979,69, WordPerSec = 864,10
Update = 10800, Epoch = 2, LR = 0,000577, AvgCost = 231,2044, Sent = 363, SentPerMin = 961,12, WordPerSec = 886,28
Update = 10900, Epoch = 2, LR = 0,000575, AvgCost = 227,7861, Sent = 656, SentPerMin = 970,98, WordPerSec = 894,01
Update = 11000, Epoch = 2, LR = 0,000572, AvgCost = 225,2132, Sent = 961, SentPerMin = 983,31, WordPerSec = 901,26
Update = 11014, Epoch = 2, LR = 0,000572, AvgCost = 225,6211, Sent = 1000, SentPerMin = 983,56, WordPerSec = 902,68
Update = 11100, Epoch = 3, LR = 0,000569, AvgCost = 229,2221, Sent = 248, SentPerMin = 970,40, WordPerSec = 896,44
Update = 11200, Epoch = 3, LR = 0,000567, AvgCost = 224,9950, Sent = 549, SentPerMin = 981,46, WordPerSec = 899,26
Update = 11300, Epoch = 3, LR = 0,000564, AvgCost = 224,8526, Sent = 841, SentPerMin = 980,93, WordPerSec = 903,29
Update = 11352, Epoch = 3, LR = 0,000563, AvgCost = 223,3539, Sent = 1000, SentPerMin = 987,15, WordPerSec = 905,97
Update = 11400, Epoch = 4, LR = 0,000562, AvgCost = 221,8619, Sent = 140, SentPerMin = 972,68, WordPerSec = 873,09
Update = 11500, Epoch = 4, LR = 0,000560, AvgCost = 228,7120, Sent = 431, SentPerMin = 958,80, WordPerSec = 888,76
Update = 11600, Epoch = 4, LR = 0,000557, AvgCost = 221,5233, Sent = 732, SentPerMin = 979,70, WordPerSec = 895,69
Update = 11690, Epoch = 4, LR = 0,000555, AvgCost = 221,5804, Sent = 1000, SentPerMin = 983,08, WordPerSec = 902,24
Update = 11700, Epoch = 5, LR = 0,000555, AvgCost = 183,8376, Sent = 33, SentPerMin = 1045,27, WordPerSec = 807,70
Update = 11800, Epoch = 5, LR = 0,000552, AvgCost = 229,0233, Sent = 317, SentPerMin = 958,69, WordPerSec = 892,76
Update = 11900, Epoch = 5, LR = 0,000550, AvgCost = 222,4954, Sent = 618, SentPerMin = 973,62, WordPerSec = 893,56
Update = 12000, Epoch = 5, LR = 0,000548, AvgCost = 221,7049, Sent = 914, SentPerMin = 978,09, WordPerSec = 901,31
Update = 12028, Epoch = 5, LR = 0,000547, AvgCost = 220,5149, Sent = 1000, SentPerMin = 983,01, WordPerSec = 902,18
Update = 12100, Epoch = 6, LR = 0,000545, AvgCost = 229,0874, Sent = 205, SentPerMin = 909,25, WordPerSec = 848,93
Update = 12200, Epoch = 6, LR = 0,000543, AvgCost = 221,7412, Sent = 506, SentPerMin = 924,13, WordPerSec = 847,64
Update = 12300, Epoch = 6, LR = 0,000541, AvgCost = 221,5887, Sent = 802, SentPerMin = 906,58, WordPerSec = 834,63
Update = 12366, Epoch = 6, LR = 0,000540, AvgCost = 219,6008, Sent = 1000, SentPerMin = 905,78, WordPerSec = 831,30
Update = 12400, Epoch = 7, LR = 0,000539, AvgCost = 209,7631, Sent = 106, SentPerMin = 942,01, WordPerSec = 811,38
Update = 12500, Epoch = 7, LR = 0,000537, AvgCost = 222,8165, Sent = 393, SentPerMin = 906,06, WordPerSec = 835,05
Update = 12600, Epoch = 7, LR = 0,000535, AvgCost = 219,6578, Sent = 689, SentPerMin = 928,85, WordPerSec = 851,44
Update = 12700, Epoch = 7, LR = 0,000532, AvgCost = 219,1764, Sent = 989, SentPerMin = 937,33, WordPerSec = 860,48
Update = 12704, Epoch = 7, LR = 0,000532, AvgCost = 219,0071, Sent = 1000, SentPerMin = 937,77, WordPerSec = 860,65
Update = 12800, Epoch = 8, LR = 0,000530, AvgCost = 229,4815, Sent = 275, SentPerMin = 941,52, WordPerSec = 883,83
Update = 12900, Epoch = 8, LR = 0,000528, AvgCost = 219,6337, Sent = 581, SentPerMin = 971,80, WordPerSec = 891,48
Update = 13000, Epoch = 8, LR = 0,000526, AvgCost = 219,9053, Sent = 870, SentPerMin = 973,11, WordPerSec = 898,45
Update = 13042, Epoch = 8, LR = 0,000525, AvgCost = 218,3436, Sent = 1000, SentPerMin = 981,34, WordPerSec = 900,64
Update = 13100, Epoch = 9, LR = 0,000524, AvgCost = 222,0313, Sent = 166, SentPerMin = 949,84, WordPerSec = 875,27
Update = 13200, Epoch = 9, LR = 0,000522, AvgCost = 221,9140, Sent = 463, SentPerMin = 965,80, WordPerSec = 891,44
Update = 13300, Epoch = 9, LR = 0,000520, AvgCost = 220,0619, Sent = 761, SentPerMin = 973,18, WordPerSec = 896,16
Update = 13380, Epoch = 9, LR = 0,000519, AvgCost = 217,9270, Sent = 1000, SentPerMin = 983,22, WordPerSec = 902,36
Update = 13400, Epoch = 10, LR = 0,000518, AvgCost = 216,0418, Sent = 64, SentPerMin = 722,53, WordPerSec = 629,95
Update = 13500, Epoch = 10, LR = 0,000516, AvgCost = 222,3316, Sent = 352, SentPerMin = 685,15, WordPerSec = 630,06
Update = 13600, Epoch = 10, LR = 0,000514, AvgCost = 219,7079, Sent = 645, SentPerMin = 684,96, WordPerSec = 630,48
Update = 13700, Epoch = 10, LR = 0,000513, AvgCost = 218,7268, Sent = 948, SentPerMin = 690,21, WordPerSec = 634,51
Update = 13718, Epoch = 10, LR = 0,000512, AvgCost = 218,2445, Sent = 1000, SentPerMin = 691,39, WordPerSec = 634,54
Update = 13800, Epoch = 11, LR = 0,000511, AvgCost = 220,0734, Sent = 239, SentPerMin = 974,27, WordPerSec = 889,75
Update = 13900, Epoch = 11, LR = 0,000509, AvgCost = 219,6372, Sent = 536, SentPerMin = 972,69, WordPerSec = 894,09
Update = 14000, Epoch = 11, LR = 0,000507, AvgCost = 218,0806, Sent = 831, SentPerMin = 978,99, WordPerSec = 898,94
Update = 14056, Epoch = 11, LR = 0,000506, AvgCost = 217,3810, Sent = 1000, SentPerMin = 983,53, WordPerSec = 902,65
Update = 14100, Epoch = 12, LR = 0,000505, AvgCost = 216,0710, Sent = 131, SentPerMin = 991,06, WordPerSec = 882,00
Update = 14200, Epoch = 12, LR = 0,000504, AvgCost = 218,9637, Sent = 423, SentPerMin = 972,27, WordPerSec = 893,51
Update = 14300, Epoch = 12, LR = 0,000502, AvgCost = 216,2776, Sent = 720, SentPerMin = 983,61, WordPerSec = 900,16
Update = 14394, Epoch = 12, LR = 0,000500, AvgCost = 216,6131, Sent = 1000, SentPerMin = 985,58, WordPerSec = 904,53
Update = 14400, Epoch = 13, LR = 0,000500, AvgCost = 216,9724, Sent = 16, SentPerMin = 891,35, WordPerSec = 869,99
Update = 14500, Epoch = 13, LR = 0,000498, AvgCost = 223,9600, Sent = 304, SentPerMin = 955,67, WordPerSec = 892,58
Update = 14600, Epoch = 13, LR = 0,000497, AvgCost = 217,4839, Sent = 606, SentPerMin = 973,48, WordPerSec = 894,69
Update = 14700, Epoch = 13, LR = 0,000495, AvgCost = 217,1494, Sent = 902, SentPerMin = 978,51, WordPerSec = 901,20
Update = 14732, Epoch = 13, LR = 0,000494, AvgCost = 215,9360, Sent = 1000, SentPerMin = 983,76, WordPerSec = 902,86
Update = 14800, Epoch = 14, LR = 0,000493, AvgCost = 220,6945, Sent = 195, SentPerMin = 964,49, WordPerSec = 886,50
Update = 14900, Epoch = 14, LR = 0,000492, AvgCost = 216,1496, Sent = 495, SentPerMin = 978,63, WordPerSec = 894,37
Update = 15000, Epoch = 14, LR = 0,000490, AvgCost = 217,3000, Sent = 789, SentPerMin = 975,69, WordPerSec = 899,66
Update = 15070, Epoch = 14, LR = 0,000489, AvgCost = 215,1631, Sent = 1000, SentPerMin = 984,78, WordPerSec = 903,80
Update = 15100, Epoch = 15, LR = 0,000488, AvgCost = 212,6104, Sent = 92, SentPerMin = 989,71, WordPerSec = 878,37
Update = 15200, Epoch = 15, LR = 0,000487, AvgCost = 217,8450, Sent = 381, SentPerMin = 965,91, WordPerSec = 888,84
Update = 15300, Epoch = 15, LR = 0,000485, AvgCost = 215,1525, Sent = 676, SentPerMin = 977,62, WordPerSec = 897,52
Update = 15400, Epoch = 15, LR = 0,000483, AvgCost = 213,1364, Sent = 979, SentPerMin = 987,03, WordPerSec = 903,78
Update = 15408, Epoch = 15, LR = 0,000483, AvgCost = 214,2401, Sent = 1000, SentPerMin = 984,90, WordPerSec = 903,90
Update = 15500, Epoch = 16, LR = 0,000482, AvgCost = 221,3075, Sent = 265, SentPerMin = 958,80, WordPerSec = 889,82
Update = 15600, Epoch = 16, LR = 0,000480, AvgCost = 215,4506, Sent = 566, SentPerMin = 972,55, WordPerSec = 895,11
Update = 15700, Epoch = 16, LR = 0,000479, AvgCost = 214,8995, Sent = 859, SentPerMin = 975,46, WordPerSec = 898,88
Update = 15746, Epoch = 16, LR = 0,000478, AvgCost = 213,5007, Sent = 1000, SentPerMin = 983,71, WordPerSec = 902,81
Update = 15800, Epoch = 17, LR = 0,000477, AvgCost = 216,4509, Sent = 157, SentPerMin = 967,40, WordPerSec = 880,01
Update = 15900, Epoch = 17, LR = 0,000476, AvgCost = 217,1673, Sent = 450, SentPerMin = 968,21, WordPerSec = 892,79
Update = 16000, Epoch = 17, LR = 0,000474, AvgCost = 212,8455, Sent = 752, SentPerMin = 978,51, WordPerSec = 897,12
Update = 16084, Epoch = 17, LR = 0,000473, AvgCost = 212,8151, Sent = 1000, SentPerMin = 982,87, WordPerSec = 902,05
Update = 16100, Epoch = 18, LR = 0,000473, AvgCost = 208,9460, Sent = 51, SentPerMin = 1013,70, WordPerSec = 873,91
Update = 16200, Epoch = 18, LR = 0,000471, AvgCost = 218,7649, Sent = 336, SentPerMin = 963,52, WordPerSec = 892,69
Update = 16300, Epoch = 18, LR = 0,000470, AvgCost = 212,9766, Sent = 636, SentPerMin = 976,24, WordPerSec = 894,45
Update = 16400, Epoch = 18, LR = 0,000469, AvgCost = 212,6840, Sent = 935, SentPerMin = 979,62, WordPerSec = 900,51
Update = 16422, Epoch = 18, LR = 0,000468, AvgCost = 212,1935, Sent = 1000, SentPerMin = 983,27, WordPerSec = 902,41
Update = 16500, Epoch = 19, LR = 0,000467, AvgCost = 214,9748, Sent = 225, SentPerMin = 965,96, WordPerSec = 887,97
Update = 16600, Epoch = 19, LR = 0,000466, AvgCost = 214,1730, Sent = 524, SentPerMin = 971,04, WordPerSec = 893,48
Update = 16700, Epoch = 19, LR = 0,000464, AvgCost = 212,6338, Sent = 819, SentPerMin = 977,04, WordPerSec = 898,68
Update = 16760, Epoch = 19, LR = 0,000463, AvgCost = 211,2749, Sent = 1000, SentPerMin = 983,79, WordPerSec = 902,89
Starting inference...
Inference results:
和 在               ) )        )      )    ) 的        ) ) 的。
的 在          )) 和        ) ) 。
和 在               )) 和     )     )) 和      )   ) 的     ) ) 的。
的 在          )) 和      )    ) ) 。
的 在           )) 和         ) ) 。

             */


            /*
             Update = 10100, Epoch = 0, LR = 0,000597, AvgCost = 267,7863, Sent = 285, SentPerMin = 99,94, WordPerSec = 93,84
Update = 10200, Epoch = 0, LR = 0,000594, AvgCost = 248,8258, Sent = 592, SentPerMin = 101,83, WordPerSec = 93,41
Update = 10300, Epoch = 0, LR = 0,000591, AvgCost = 246,6337, Sent = 882, SentPerMin = 102,39, WordPerSec = 94,59
Update = 10338, Epoch = 0, LR = 0,000590, AvgCost = 243,4159, Sent = 1000, SentPerMin = 103,29, WordPerSec = 94,79
Update = 10400, Epoch = 1, LR = 0,000588, AvgCost = 233,9075, Sent = 178, SentPerMin = 142,18, WordPerSec = 131,19
Update = 10500, Epoch = 1, LR = 0,000586, AvgCost = 234,3407, Sent = 475, SentPerMin = 142,02, WordPerSec = 130,81
Update = 10600, Epoch = 1, LR = 0,000583, AvgCost = 232,6985, Sent = 771, SentPerMin = 142,19, WordPerSec = 131,42
Update = 10676, Epoch = 1, LR = 0,000581, AvgCost = 229,4459, Sent = 1000, SentPerMin = 144,39, WordPerSec = 132,52
Update = 10700, Epoch = 2, LR = 0,000580, AvgCost = 220,2904, Sent = 76, SentPerMin = 103,64, WordPerSec = 91,41
Update = 10800, Epoch = 2, LR = 0,000577, AvgCost = 230,8826, Sent = 363, SentPerMin = 102,51, WordPerSec = 94,53
Update = 10900, Epoch = 2, LR = 0,000575, AvgCost = 227,6483, Sent = 656, SentPerMin = 103,14, WordPerSec = 94,97
Update = 11000, Epoch = 2, LR = 0,000572, AvgCost = 225,1791, Sent = 961, SentPerMin = 103,92, WordPerSec = 95,25
Update = 11014, Epoch = 2, LR = 0,000572, AvgCost = 225,5483, Sent = 1000, SentPerMin = 104,04, WordPerSec = 95,48
Update = 11100, Epoch = 3, LR = 0,000569, AvgCost = 228,7370, Sent = 248, SentPerMin = 143,85, WordPerSec = 132,89
Update = 11200, Epoch = 3, LR = 0,000567, AvgCost = 224,7752, Sent = 549, SentPerMin = 143,01, WordPerSec = 131,03
Update = 11300, Epoch = 3, LR = 0,000564, AvgCost = 224,6235, Sent = 841, SentPerMin = 143,22, WordPerSec = 131,88
Update = 11352, Epoch = 3, LR = 0,000563, AvgCost = 223,1907, Sent = 1000, SentPerMin = 144,18, WordPerSec = 132,33
Update = 11400, Epoch = 4, LR = 0,000562, AvgCost = 221,1435, Sent = 140, SentPerMin = 105,50, WordPerSec = 94,69
Update = 11500, Epoch = 4, LR = 0,000560, AvgCost = 227,9353, Sent = 431, SentPerMin = 102,25, WordPerSec = 94,78
Update = 11600, Epoch = 4, LR = 0,000557, AvgCost = 221,0524, Sent = 732, SentPerMin = 104,10, WordPerSec = 95,17
Update = 11690, Epoch = 4, LR = 0,000555, AvgCost = 221,4366, Sent = 1000, SentPerMin = 104,05, WordPerSec = 95,49
Update = 11700, Epoch = 5, LR = 0,000555, AvgCost = 182,4858, Sent = 33, SentPerMin = 159,71, WordPerSec = 123,41
Update = 11800, Epoch = 5, LR = 0,000552, AvgCost = 227,5006, Sent = 317, SentPerMin = 142,00, WordPerSec = 132,24
Update = 11900, Epoch = 5, LR = 0,000550, AvgCost = 221,4011, Sent = 618, SentPerMin = 142,92, WordPerSec = 131,17
Update = 12000, Epoch = 5, LR = 0,000548, AvgCost = 221,2347, Sent = 914, SentPerMin = 143,38, WordPerSec = 132,12
Update = 12028, Epoch = 5, LR = 0,000547, AvgCost = 220,0866, Sent = 1000, SentPerMin = 144,28, WordPerSec = 132,42
Update = 12100, Epoch = 6, LR = 0,000545, AvgCost = 227,4382, Sent = 205, SentPerMin = 102,32, WordPerSec = 95,53
Update = 12200, Epoch = 6, LR = 0,000543, AvgCost = 220,8730, Sent = 506, SentPerMin = 102,81, WordPerSec = 94,30
Update = 12300, Epoch = 6, LR = 0,000541, AvgCost = 221,0429, Sent = 802, SentPerMin = 103,22, WordPerSec = 95,03
Update = 12366, Epoch = 6, LR = 0,000540, AvgCost = 219,3626, Sent = 1000, SentPerMin = 104,03, WordPerSec = 95,48
Update = 12400, Epoch = 7, LR = 0,000539, AvgCost = 208,6927, Sent = 106, SentPerMin = 146,97, WordPerSec = 126,59
Update = 12500, Epoch = 7, LR = 0,000537, AvgCost = 221,8916, Sent = 393, SentPerMin = 141,23, WordPerSec = 130,16
Update = 12600, Epoch = 7, LR = 0,000535, AvgCost = 218,5168, Sent = 689, SentPerMin = 144,38, WordPerSec = 132,35
Update = 12700, Epoch = 7, LR = 0,000532, AvgCost = 218,3349, Sent = 989, SentPerMin = 144,18, WordPerSec = 132,36
Update = 12704, Epoch = 7, LR = 0,000532, AvgCost = 218,1845, Sent = 1000, SentPerMin = 144,30, WordPerSec = 132,43
Update = 12800, Epoch = 8, LR = 0,000530, AvgCost = 228,9367, Sent = 275, SentPerMin = 102,00, WordPerSec = 95,75
Update = 12900, Epoch = 8, LR = 0,000528, AvgCost = 218,7443, Sent = 581, SentPerMin = 102,65, WordPerSec = 94,17
Update = 13000, Epoch = 8, LR = 0,000526, AvgCost = 219,0822, Sent = 870, SentPerMin = 103,32, WordPerSec = 95,39
Update = 13042, Epoch = 8, LR = 0,000525, AvgCost = 217,5796, Sent = 1000, SentPerMin = 103,95, WordPerSec = 95,40
Update = 13100, Epoch = 9, LR = 0,000524, AvgCost = 221,0322, Sent = 166, SentPerMin = 142,09, WordPerSec = 130,93
Update = 13200, Epoch = 9, LR = 0,000522, AvgCost = 221,0346, Sent = 463, SentPerMin = 141,36, WordPerSec = 130,48
Update = 13300, Epoch = 9, LR = 0,000520, AvgCost = 218,7567, Sent = 761, SentPerMin = 142,04, WordPerSec = 130,80
Update = 13380, Epoch = 9, LR = 0,000519, AvgCost = 216,5574, Sent = 1000, SentPerMin = 144,09, WordPerSec = 132,24
Update = 13400, Epoch = 10, LR = 0,000518, AvgCost = 211,9049, Sent = 64, SentPerMin = 106,23, WordPerSec = 92,62
Update = 13500, Epoch = 10, LR = 0,000516, AvgCost = 220,5992, Sent = 352, SentPerMin = 102,42, WordPerSec = 94,19
Update = 13600, Epoch = 10, LR = 0,000514, AvgCost = 217,1538, Sent = 645, SentPerMin = 102,87, WordPerSec = 94,69
Update = 13700, Epoch = 10, LR = 0,000513, AvgCost = 216,0126, Sent = 948, SentPerMin = 103,33, WordPerSec = 94,99
Update = 13718, Epoch = 10, LR = 0,000512, AvgCost = 215,5145, Sent = 1000, SentPerMin = 103,69, WordPerSec = 95,16
Update = 13800, Epoch = 11, LR = 0,000511, AvgCost = 217,3917, Sent = 239, SentPerMin = 145,36, WordPerSec = 132,75
Update = 13900, Epoch = 11, LR = 0,000509, AvgCost = 216,7065, Sent = 536, SentPerMin = 142,18, WordPerSec = 130,69
Update = 14000, Epoch = 11, LR = 0,000507, AvgCost = 215,1359, Sent = 831, SentPerMin = 143,22, WordPerSec = 131,51
Update = 14056, Epoch = 11, LR = 0,000506, AvgCost = 214,2685, Sent = 1000, SentPerMin = 143,96, WordPerSec = 132,12
Update = 14100, Epoch = 12, LR = 0,000505, AvgCost = 213,1106, Sent = 131, SentPerMin = 105,62, WordPerSec = 94,00
Update = 14200, Epoch = 12, LR = 0,000504, AvgCost = 217,4739, Sent = 423, SentPerMin = 102,48, WordPerSec = 94,17
Update = 14300, Epoch = 12, LR = 0,000502, AvgCost = 214,0921, Sent = 720, SentPerMin = 103,84, WordPerSec = 95,03
Update = 14394, Epoch = 12, LR = 0,000500, AvgCost = 214,2203, Sent = 1000, SentPerMin = 103,83, WordPerSec = 95,29
Update = 14400, Epoch = 13, LR = 0,000500, AvgCost = 212,4752, Sent = 16, SentPerMin = 141,84, WordPerSec = 138,44
Update = 14500, Epoch = 13, LR = 0,000498, AvgCost = 221,9901, Sent = 304, SentPerMin = 142,23, WordPerSec = 132,84
Update = 14600, Epoch = 13, LR = 0,000497, AvgCost = 214,6003, Sent = 606, SentPerMin = 142,43, WordPerSec = 130,90
Update = 14700, Epoch = 13, LR = 0,000495, AvgCost = 214,2569, Sent = 902, SentPerMin = 143,16, WordPerSec = 131,85
Update = 14732, Epoch = 13, LR = 0,000494, AvgCost = 213,0381, Sent = 1000, SentPerMin = 144,09, WordPerSec = 132,24
Update = 14800, Epoch = 14, LR = 0,000493, AvgCost = 219,0377, Sent = 195, SentPerMin = 103,66, WordPerSec = 95,28
Update = 14900, Epoch = 14, LR = 0,000492, AvgCost = 214,5523, Sent = 495, SentPerMin = 103,27, WordPerSec = 94,38
Update = 15000, Epoch = 14, LR = 0,000490, AvgCost = 215,1722, Sent = 789, SentPerMin = 102,67, WordPerSec = 94,67
Update = 15070, Epoch = 14, LR = 0,000489, AvgCost = 212,7598, Sent = 1000, SentPerMin = 103,72, WordPerSec = 95,19
Update = 15100, Epoch = 15, LR = 0,000488, AvgCost = 209,4172, Sent = 92, SentPerMin = 142,02, WordPerSec = 126,05
Update = 15200, Epoch = 15, LR = 0,000487, AvgCost = 215,5969, Sent = 381, SentPerMin = 141,07, WordPerSec = 129,81
Update = 15300, Epoch = 15, LR = 0,000485, AvgCost = 212,5274, Sent = 676, SentPerMin = 143,23, WordPerSec = 131,49
Update = 15400, Epoch = 15, LR = 0,000483, AvgCost = 210,1993, Sent = 979, SentPerMin = 144,20, WordPerSec = 132,04
Update = 15408, Epoch = 15, LR = 0,000483, AvgCost = 211,3229, Sent = 1000, SentPerMin = 144,01, WordPerSec = 132,17
Update = 15500, Epoch = 16, LR = 0,000482, AvgCost = 219,1265, Sent = 265, SentPerMin = 102,90, WordPerSec = 95,50
Update = 15600, Epoch = 16, LR = 0,000480, AvgCost = 212,9533, Sent = 566, SentPerMin = 102,62, WordPerSec = 94,45
Update = 15700, Epoch = 16, LR = 0,000479, AvgCost = 212,2545, Sent = 859, SentPerMin = 102,95, WordPerSec = 94,87
Update = 15746, Epoch = 16, LR = 0,000478, AvgCost = 211,0098, Sent = 1000, SentPerMin = 103,69, WordPerSec = 95,16
Update = 15800, Epoch = 17, LR = 0,000477, AvgCost = 212,4074, Sent = 157, SentPerMin = 142,59, WordPerSec = 129,71
Update = 15900, Epoch = 17, LR = 0,000476, AvgCost = 214,1439, Sent = 450, SentPerMin = 141,67, WordPerSec = 130,64
Update = 16000, Epoch = 17, LR = 0,000474, AvgCost = 209,6181, Sent = 752, SentPerMin = 142,52, WordPerSec = 130,66
Update = 16084, Epoch = 17, LR = 0,000473, AvgCost = 209,6444, Sent = 1000, SentPerMin = 143,95, WordPerSec = 132,11
Update = 16100, Epoch = 18, LR = 0,000473, AvgCost = 203,1104, Sent = 51, SentPerMin = 108,25, WordPerSec = 93,32
Update = 16200, Epoch = 18, LR = 0,000471, AvgCost = 215,1924, Sent = 336, SentPerMin = 102,75, WordPerSec = 95,19
Update = 16300, Epoch = 18, LR = 0,000470, AvgCost = 209,4279, Sent = 636, SentPerMin = 103,10, WordPerSec = 94,47
Update = 16400, Epoch = 18, LR = 0,000469, AvgCost = 209,3707, Sent = 935, SentPerMin = 102,71, WordPerSec = 94,42
Update = 16422, Epoch = 18, LR = 0,000468, AvgCost = 209,0042, Sent = 1000, SentPerMin = 103,24, WordPerSec = 94,75
Update = 16500, Epoch = 19, LR = 0,000467, AvgCost = 211,6619, Sent = 225, SentPerMin = 144,71, WordPerSec = 133,03
Update = 16600, Epoch = 19, LR = 0,000466, AvgCost = 210,6525, Sent = 524, SentPerMin = 141,62, WordPerSec = 130,31
Update = 16700, Epoch = 19, LR = 0,000464, AvgCost = 208,8903, Sent = 819, SentPerMin = 141,85, WordPerSec = 130,47
Update = 16760, Epoch = 19, LR = 0,000463, AvgCost = 207,6941, Sent = 1000, SentPerMin = 142,81, WordPerSec = 131,07
Update = 16800, Epoch = 20, LR = 0,000463, AvgCost = 204,0432, Sent = 122, SentPerMin = 104,47, WordPerSec = 91,18
Update = 16900, Epoch = 20, LR = 0,000462, AvgCost = 211,2490, Sent = 411, SentPerMin = 100,97, WordPerSec = 92,99
Update = 17000, Epoch = 20, LR = 0,000460, AvgCost = 207,1805, Sent = 709, SentPerMin = 103,41, WordPerSec = 94,46
Update = 17098, Epoch = 20, LR = 0,000459, AvgCost = 207,8149, Sent = 1000, SentPerMin = 103,26, WordPerSec = 94,77
Update = 17100, Epoch = 21, LR = 0,000459, AvgCost = 182,6849, Sent = 6, SentPerMin = 156,04, WordPerSec = 141,74
Update = 17200, Epoch = 21, LR = 0,000457, AvgCost = 215,7098, Sent = 292, SentPerMin = 141,69, WordPerSec = 132,57
Update = 17300, Epoch = 21, LR = 0,000456, AvgCost = 208,3996, Sent = 595, SentPerMin = 142,00, WordPerSec = 130,70
Update = 17400, Epoch = 21, LR = 0,000455, AvgCost = 207,7799, Sent = 890, SentPerMin = 142,88, WordPerSec = 131,71
Update = 17436, Epoch = 21, LR = 0,000454, AvgCost = 206,4537, Sent = 1000, SentPerMin = 143,87, WordPerSec = 132,04
Update = 17500, Epoch = 22, LR = 0,000454, AvgCost = 212,6966, Sent = 184, SentPerMin = 103,50, WordPerSec = 95,09
Update = 17600, Epoch = 22, LR = 0,000452, AvgCost = 208,9788, Sent = 482, SentPerMin = 102,51, WordPerSec = 94,24
Update = 17700, Epoch = 22, LR = 0,000451, AvgCost = 209,4970, Sent = 775, SentPerMin = 102,03, WordPerSec = 94,54
Update = 17774, Epoch = 22, LR = 0,000450, AvgCost = 206,4982, Sent = 1000, SentPerMin = 103,62, WordPerSec = 95,10
Update = 17800, Epoch = 23, LR = 0,000450, AvgCost = 200,0937, Sent = 81, SentPerMin = 141,96, WordPerSec = 126,01
Update = 17900, Epoch = 23, LR = 0,000448, AvgCost = 210,5154, Sent = 368, SentPerMin = 140,17, WordPerSec = 129,72
Update = 18000, Epoch = 23, LR = 0,000447, AvgCost = 206,2954, Sent = 663, SentPerMin = 142,82, WordPerSec = 131,22
Update = 18100, Epoch = 23, LR = 0,000446, AvgCost = 204,3346, Sent = 969, SentPerMin = 143,99, WordPerSec = 131,67
Update = 18112, Epoch = 23, LR = 0,000446, AvgCost = 205,2653, Sent = 1000, SentPerMin = 143,91, WordPerSec = 132,07
Update = 18200, Epoch = 24, LR = 0,000445, AvgCost = 215,6516, Sent = 251, SentPerMin = 102,47, WordPerSec = 95,73
Update = 18300, Epoch = 24, LR = 0,000444, AvgCost = 206,2585, Sent = 556, SentPerMin = 103,59, WordPerSec = 94,82
Update = 18400, Epoch = 24, LR = 0,000442, AvgCost = 206,8590, Sent = 847, SentPerMin = 103,37, WordPerSec = 95,28
Update = 18450, Epoch = 24, LR = 0,000442, AvgCost = 205,6482, Sent = 1000, SentPerMin = 103,93, WordPerSec = 95,38
Update = 18500, Epoch = 25, LR = 0,000441, AvgCost = 207,4532, Sent = 145, SentPerMin = 142,05, WordPerSec = 129,20
Update = 18600, Epoch = 25, LR = 0,000440, AvgCost = 209,8481, Sent = 437, SentPerMin = 141,24, WordPerSec = 130,88
Update = 18700, Epoch = 25, LR = 0,000439, AvgCost = 203,9675, Sent = 739, SentPerMin = 142,71, WordPerSec = 130,82
Update = 18788, Epoch = 25, LR = 0,000438, AvgCost = 204,2488, Sent = 1000, SentPerMin = 143,99, WordPerSec = 132,15
Starting inference...
Inference results:
,•的的                                  。
,•的在                 。
• ( ( ( ()) 的
• ( ( ( ()) ) 的
• ( ( () ) 的


             */


            /*
             Update = 10100, Epoch = 0, LR = 0,000597, AvgCost = 266,7863, Sent = 285, SentPerMin = 100,40, WordPerSec = 94,28
Update = 10200, Epoch = 0, LR = 0,000594, AvgCost = 248,1894, Sent = 592, SentPerMin = 102,03, WordPerSec = 93,60
Update = 10300, Epoch = 0, LR = 0,000591, AvgCost = 245,9188, Sent = 882, SentPerMin = 102,48, WordPerSec = 94,68
Update = 10338, Epoch = 0, LR = 0,000590, AvgCost = 242,7421, Sent = 1000, SentPerMin = 103,32, WordPerSec = 94,83
Update = 10400, Epoch = 1, LR = 0,000588, AvgCost = 234,4302, Sent = 178, SentPerMin = 142,17, WordPerSec = 131,19
Update = 10500, Epoch = 1, LR = 0,000586, AvgCost = 234,5072, Sent = 475, SentPerMin = 142,00, WordPerSec = 130,80
Update = 10600, Epoch = 1, LR = 0,000583, AvgCost = 232,8764, Sent = 771, SentPerMin = 141,96, WordPerSec = 131,22
Update = 10676, Epoch = 1, LR = 0,000581, AvgCost = 229,5788, Sent = 1000, SentPerMin = 144,16, WordPerSec = 132,31
Update = 10700, Epoch = 2, LR = 0,000580, AvgCost = 220,4875, Sent = 76, SentPerMin = 103,16, WordPerSec = 90,99
Update = 10800, Epoch = 2, LR = 0,000577, AvgCost = 230,6244, Sent = 363, SentPerMin = 102,19, WordPerSec = 94,23
Update = 10900, Epoch = 2, LR = 0,000575, AvgCost = 227,3322, Sent = 656, SentPerMin = 102,94, WordPerSec = 94,78
Update = 11000, Epoch = 2, LR = 0,000572, AvgCost = 224,8656, Sent = 961, SentPerMin = 103,71, WordPerSec = 95,06
Update = 11014, Epoch = 2, LR = 0,000572, AvgCost = 225,2559, Sent = 1000, SentPerMin = 103,83, WordPerSec = 95,29
Update = 11100, Epoch = 3, LR = 0,000569, AvgCost = 228,5373, Sent = 248, SentPerMin = 143,35, WordPerSec = 132,43
Update = 11200, Epoch = 3, LR = 0,000567, AvgCost = 224,4849, Sent = 549, SentPerMin = 142,66, WordPerSec = 130,71
Update = 11300, Epoch = 3, LR = 0,000564, AvgCost = 224,5344, Sent = 841, SentPerMin = 143,03, WordPerSec = 131,71
Update = 11352, Epoch = 3, LR = 0,000563, AvgCost = 223,1461, Sent = 1000, SentPerMin = 144,08, WordPerSec = 132,23
Update = 11400, Epoch = 4, LR = 0,000562, AvgCost = 220,1071, Sent = 140, SentPerMin = 105,48, WordPerSec = 94,68
Update = 11500, Epoch = 4, LR = 0,000560, AvgCost = 227,7459, Sent = 431, SentPerMin = 102,21, WordPerSec = 94,75
Update = 11600, Epoch = 4, LR = 0,000557, AvgCost = 220,8597, Sent = 732, SentPerMin = 103,96, WordPerSec = 95,05
Update = 11690, Epoch = 4, LR = 0,000555, AvgCost = 221,0659, Sent = 1000, SentPerMin = 103,91, WordPerSec = 95,36
Update = 11700, Epoch = 5, LR = 0,000555, AvgCost = 182,1951, Sent = 33, SentPerMin = 159,02, WordPerSec = 122,88
Update = 11800, Epoch = 5, LR = 0,000552, AvgCost = 227,1780, Sent = 317, SentPerMin = 141,86, WordPerSec = 132,10
Update = 11900, Epoch = 5, LR = 0,000550, AvgCost = 221,5123, Sent = 618, SentPerMin = 142,99, WordPerSec = 131,23
Update = 12000, Epoch = 5, LR = 0,000548, AvgCost = 220,9157, Sent = 914, SentPerMin = 143,40, WordPerSec = 132,14
Update = 12028, Epoch = 5, LR = 0,000547, AvgCost = 219,7206, Sent = 1000, SentPerMin = 144,32, WordPerSec = 132,45
Update = 12100, Epoch = 6, LR = 0,000545, AvgCost = 227,2277, Sent = 205, SentPerMin = 102,20, WordPerSec = 95,42
Update = 12200, Epoch = 6, LR = 0,000543, AvgCost = 220,7590, Sent = 506, SentPerMin = 102,65, WordPerSec = 94,16
Update = 12300, Epoch = 6, LR = 0,000541, AvgCost = 220,8150, Sent = 802, SentPerMin = 103,07, WordPerSec = 94,89
Update = 12366, Epoch = 6, LR = 0,000540, AvgCost = 218,8840, Sent = 1000, SentPerMin = 103,91, WordPerSec = 95,37
Update = 12400, Epoch = 7, LR = 0,000539, AvgCost = 208,3681, Sent = 106, SentPerMin = 146,14, WordPerSec = 125,87
Update = 12500, Epoch = 7, LR = 0,000537, AvgCost = 221,3555, Sent = 393, SentPerMin = 140,84, WordPerSec = 129,80
Update = 12600, Epoch = 7, LR = 0,000535, AvgCost = 218,3502, Sent = 689, SentPerMin = 144,06, WordPerSec = 132,05
Update = 12700, Epoch = 7, LR = 0,000532, AvgCost = 217,9152, Sent = 989, SentPerMin = 144,00, WordPerSec = 132,20
Update = 12704, Epoch = 7, LR = 0,000532, AvgCost = 217,7601, Sent = 1000, SentPerMin = 144,12, WordPerSec = 132,27
Update = 12800, Epoch = 8, LR = 0,000530, AvgCost = 228,7049, Sent = 275, SentPerMin = 101,97, WordPerSec = 95,72
Update = 12900, Epoch = 8, LR = 0,000528, AvgCost = 218,6346, Sent = 581, SentPerMin = 102,63, WordPerSec = 94,15
Update = 13000, Epoch = 8, LR = 0,000526, AvgCost = 218,8978, Sent = 870, SentPerMin = 103,25, WordPerSec = 95,32
Update = 13042, Epoch = 8, LR = 0,000525, AvgCost = 217,4275, Sent = 1000, SentPerMin = 103,90, WordPerSec = 95,36
Update = 13100, Epoch = 9, LR = 0,000524, AvgCost = 220,2883, Sent = 166, SentPerMin = 141,91, WordPerSec = 130,77
Update = 13200, Epoch = 9, LR = 0,000522, AvgCost = 220,4341, Sent = 463, SentPerMin = 141,28, WordPerSec = 130,40
Update = 13300, Epoch = 9, LR = 0,000520, AvgCost = 218,1903, Sent = 761, SentPerMin = 141,91, WordPerSec = 130,67
Update = 13380, Epoch = 9, LR = 0,000519, AvgCost = 216,1807, Sent = 1000, SentPerMin = 144,04, WordPerSec = 132,19
Update = 13400, Epoch = 10, LR = 0,000518, AvgCost = 212,8565, Sent = 64, SentPerMin = 106,58, WordPerSec = 92,93
Update = 13500, Epoch = 10, LR = 0,000516, AvgCost = 220,7676, Sent = 352, SentPerMin = 102,43, WordPerSec = 94,19
Update = 13600, Epoch = 10, LR = 0,000514, AvgCost = 217,5365, Sent = 645, SentPerMin = 102,87, WordPerSec = 94,68
Update = 13700, Epoch = 10, LR = 0,000513, AvgCost = 216,2085, Sent = 948, SentPerMin = 103,43, WordPerSec = 95,08
Update = 13718, Epoch = 10, LR = 0,000512, AvgCost = 215,7744, Sent = 1000, SentPerMin = 103,79, WordPerSec = 95,26
Update = 13800, Epoch = 11, LR = 0,000511, AvgCost = 217,6626, Sent = 239, SentPerMin = 145,36, WordPerSec = 132,75
Update = 13900, Epoch = 11, LR = 0,000509, AvgCost = 217,0868, Sent = 536, SentPerMin = 142,17, WordPerSec = 130,68
Update = 14000, Epoch = 11, LR = 0,000507, AvgCost = 215,2554, Sent = 831, SentPerMin = 143,23, WordPerSec = 131,52
Update = 14056, Epoch = 11, LR = 0,000506, AvgCost = 214,4995, Sent = 1000, SentPerMin = 144,02, WordPerSec = 132,18
Update = 14100, Epoch = 12, LR = 0,000505, AvgCost = 213,3534, Sent = 131, SentPerMin = 105,63, WordPerSec = 94,00
Update = 14200, Epoch = 12, LR = 0,000504, AvgCost = 216,7738, Sent = 423, SentPerMin = 102,37, WordPerSec = 94,08
Update = 14300, Epoch = 12, LR = 0,000502, AvgCost = 213,5431, Sent = 720, SentPerMin = 103,73, WordPerSec = 94,93
Update = 14394, Epoch = 12, LR = 0,000500, AvgCost = 213,7269, Sent = 1000, SentPerMin = 103,71, WordPerSec = 95,18
Update = 14400, Epoch = 13, LR = 0,000500, AvgCost = 213,3708, Sent = 16, SentPerMin = 140,61, WordPerSec = 137,24
Update = 14500, Epoch = 13, LR = 0,000498, AvgCost = 221,0698, Sent = 304, SentPerMin = 142,28, WordPerSec = 132,89
Update = 14600, Epoch = 13, LR = 0,000497, AvgCost = 214,2334, Sent = 606, SentPerMin = 142,26, WordPerSec = 130,74
Update = 14700, Epoch = 13, LR = 0,000495, AvgCost = 213,5688, Sent = 902, SentPerMin = 143,12, WordPerSec = 131,81
Update = 14732, Epoch = 13, LR = 0,000494, AvgCost = 212,3201, Sent = 1000, SentPerMin = 144,12, WordPerSec = 132,27
Update = 14800, Epoch = 14, LR = 0,000493, AvgCost = 217,3447, Sent = 195, SentPerMin = 103,54, WordPerSec = 95,17
Update = 14900, Epoch = 14, LR = 0,000492, AvgCost = 213,2146, Sent = 495, SentPerMin = 103,34, WordPerSec = 94,44
Update = 15000, Epoch = 14, LR = 0,000490, AvgCost = 213,9384, Sent = 789, SentPerMin = 102,65, WordPerSec = 94,65
Update = 15070, Epoch = 14, LR = 0,000489, AvgCost = 211,5314, Sent = 1000, SentPerMin = 103,77, WordPerSec = 95,23
Update = 15100, Epoch = 15, LR = 0,000488, AvgCost = 208,2715, Sent = 92, SentPerMin = 142,07, WordPerSec = 126,08
Update = 15200, Epoch = 15, LR = 0,000487, AvgCost = 213,8336, Sent = 381, SentPerMin = 141,24, WordPerSec = 129,97
Update = 15300, Epoch = 15, LR = 0,000485, AvgCost = 211,1804, Sent = 676, SentPerMin = 143,31, WordPerSec = 131,57
Update = 15400, Epoch = 15, LR = 0,000483, AvgCost = 208,9812, Sent = 979, SentPerMin = 144,16, WordPerSec = 132,01
Update = 15408, Epoch = 15, LR = 0,000483, AvgCost = 210,0186, Sent = 1000, SentPerMin = 143,97, WordPerSec = 132,13
Update = 15500, Epoch = 16, LR = 0,000482, AvgCost = 217,4617, Sent = 265, SentPerMin = 103,08, WordPerSec = 95,66
Update = 15600, Epoch = 16, LR = 0,000480, AvgCost = 211,8627, Sent = 566, SentPerMin = 102,80, WordPerSec = 94,61
Update = 15700, Epoch = 16, LR = 0,000479, AvgCost = 211,0484, Sent = 859, SentPerMin = 103,07, WordPerSec = 94,98
Update = 15746, Epoch = 16, LR = 0,000478, AvgCost = 209,6267, Sent = 1000, SentPerMin = 103,77, WordPerSec = 95,23
Update = 15800, Epoch = 17, LR = 0,000477, AvgCost = 211,1307, Sent = 157, SentPerMin = 142,88, WordPerSec = 129,97
Update = 15900, Epoch = 17, LR = 0,000476, AvgCost = 213,0831, Sent = 450, SentPerMin = 141,81, WordPerSec = 130,76
Update = 16000, Epoch = 17, LR = 0,000474, AvgCost = 208,2882, Sent = 752, SentPerMin = 142,64, WordPerSec = 130,77
Update = 16084, Epoch = 17, LR = 0,000473, AvgCost = 208,2294, Sent = 1000, SentPerMin = 144,10, WordPerSec = 132,25
Update = 16100, Epoch = 18, LR = 0,000473, AvgCost = 200,9786, Sent = 51, SentPerMin = 107,66, WordPerSec = 92,81
Update = 16200, Epoch = 18, LR = 0,000471, AvgCost = 214,5782, Sent = 336, SentPerMin = 102,45, WordPerSec = 94,92
Update = 16300, Epoch = 18, LR = 0,000470, AvgCost = 208,7961, Sent = 636, SentPerMin = 102,87, WordPerSec = 94,25
Update = 16400, Epoch = 18, LR = 0,000469, AvgCost = 208,7091, Sent = 935, SentPerMin = 103,14, WordPerSec = 94,81
Update = 16422, Epoch = 18, LR = 0,000468, AvgCost = 208,2288, Sent = 1000, SentPerMin = 103,65, WordPerSec = 95,13
Update = 16500, Epoch = 19, LR = 0,000467, AvgCost = 211,2504, Sent = 225, SentPerMin = 144,55, WordPerSec = 132,88
Update = 16600, Epoch = 19, LR = 0,000466, AvgCost = 210,1248, Sent = 524, SentPerMin = 141,91, WordPerSec = 130,58
Update = 16700, Epoch = 19, LR = 0,000464, AvgCost = 208,3937, Sent = 819, SentPerMin = 142,86, WordPerSec = 131,40
Update = 16760, Epoch = 19, LR = 0,000463, AvgCost = 207,1119, Sent = 1000, SentPerMin = 144,00, WordPerSec = 132,16
Update = 16800, Epoch = 20, LR = 0,000463, AvgCost = 203,3847, Sent = 122, SentPerMin = 106,49, WordPerSec = 92,95
Update = 16900, Epoch = 20, LR = 0,000462, AvgCost = 210,6781, Sent = 411, SentPerMin = 102,00, WordPerSec = 93,95
Update = 17000, Epoch = 20, LR = 0,000460, AvgCost = 206,4805, Sent = 709, SentPerMin = 104,04, WordPerSec = 95,04
Update = 17098, Epoch = 20, LR = 0,000459, AvgCost = 206,9331, Sent = 1000, SentPerMin = 103,68, WordPerSec = 95,15
Update = 17100, Epoch = 21, LR = 0,000459, AvgCost = 183,5944, Sent = 6, SentPerMin = 157,70, WordPerSec = 143,24
Update = 17200, Epoch = 21, LR = 0,000457, AvgCost = 215,6761, Sent = 292, SentPerMin = 141,49, WordPerSec = 132,38
Update = 17300, Epoch = 21, LR = 0,000456, AvgCost = 208,1752, Sent = 595, SentPerMin = 141,80, WordPerSec = 130,51
Update = 17400, Epoch = 21, LR = 0,000455, AvgCost = 207,2041, Sent = 890, SentPerMin = 142,77, WordPerSec = 131,61
Update = 17436, Epoch = 21, LR = 0,000454, AvgCost = 205,7823, Sent = 1000, SentPerMin = 143,77, WordPerSec = 131,95
Update = 17500, Epoch = 22, LR = 0,000454, AvgCost = 212,5326, Sent = 184, SentPerMin = 103,62, WordPerSec = 95,20
Update = 17600, Epoch = 22, LR = 0,000452, AvgCost = 209,2483, Sent = 482, SentPerMin = 102,66, WordPerSec = 94,38
Update = 17700, Epoch = 22, LR = 0,000451, AvgCost = 208,9146, Sent = 775, SentPerMin = 102,11, WordPerSec = 94,61
Update = 17774, Epoch = 22, LR = 0,000450, AvgCost = 205,7681, Sent = 1000, SentPerMin = 103,62, WordPerSec = 95,10
Update = 17800, Epoch = 23, LR = 0,000450, AvgCost = 199,1555, Sent = 81, SentPerMin = 141,68, WordPerSec = 125,77
Update = 17900, Epoch = 23, LR = 0,000448, AvgCost = 210,1170, Sent = 368, SentPerMin = 140,22, WordPerSec = 129,77
Update = 18000, Epoch = 23, LR = 0,000447, AvgCost = 205,8008, Sent = 663, SentPerMin = 142,85, WordPerSec = 131,25
Update = 18100, Epoch = 23, LR = 0,000446, AvgCost = 203,4876, Sent = 969, SentPerMin = 143,92, WordPerSec = 131,61
Update = 18112, Epoch = 23, LR = 0,000446, AvgCost = 204,4053, Sent = 1000, SentPerMin = 143,83, WordPerSec = 132,00
Update = 18200, Epoch = 24, LR = 0,000445, AvgCost = 214,4406, Sent = 251, SentPerMin = 102,51, WordPerSec = 95,77
Update = 18300, Epoch = 24, LR = 0,000444, AvgCost = 205,5609, Sent = 556, SentPerMin = 103,30, WordPerSec = 94,55
Update = 18400, Epoch = 24, LR = 0,000442, AvgCost = 205,5892, Sent = 847, SentPerMin = 103,00, WordPerSec = 94,93
Update = 18450, Epoch = 24, LR = 0,000442, AvgCost = 204,3459, Sent = 1000, SentPerMin = 103,62, WordPerSec = 95,10
Update = 18500, Epoch = 25, LR = 0,000441, AvgCost = 205,8702, Sent = 145, SentPerMin = 142,07, WordPerSec = 129,22
Update = 18600, Epoch = 25, LR = 0,000440, AvgCost = 209,1831, Sent = 437, SentPerMin = 141,09, WordPerSec = 130,75
Update = 18700, Epoch = 25, LR = 0,000439, AvgCost = 202,9721, Sent = 739, SentPerMin = 142,56, WordPerSec = 130,68
Update = 18788, Epoch = 25, LR = 0,000438, AvgCost = 203,0230, Sent = 1000, SentPerMin = 143,91, WordPerSec = 132,08
Update = 18800, Epoch = 26, LR = 0,000438, AvgCost = 165,1217, Sent = 41, SentPerMin = 113,72, WordPerSec = 87,55
Update = 18900, Epoch = 26, LR = 0,000436, AvgCost = 209,7298, Sent = 324, SentPerMin = 102,48, WordPerSec = 95,19
Update = 19000, Epoch = 26, LR = 0,000435, AvgCost = 204,4502, Sent = 624, SentPerMin = 102,89, WordPerSec = 94,33
Update = 19100, Epoch = 26, LR = 0,000434, AvgCost = 203,6959, Sent = 921, SentPerMin = 103,36, WordPerSec = 95,10
Update = 19126, Epoch = 26, LR = 0,000434, AvgCost = 202,9324, Sent = 1000, SentPerMin = 103,66, WordPerSec = 95,14
Update = 19200, Epoch = 27, LR = 0,000433, AvgCost = 207,1057, Sent = 213, SentPerMin = 143,08, WordPerSec = 132,00
Update = 19300, Epoch = 27, LR = 0,000432, AvgCost = 205,4839, Sent = 511, SentPerMin = 140,71, WordPerSec = 129,75
Update = 19400, Epoch = 27, LR = 0,000431, AvgCost = 203,4290, Sent = 808, SentPerMin = 142,52, WordPerSec = 131,28
Update = 19464, Epoch = 27, LR = 0,000430, AvgCost = 201,6225, Sent = 1000, SentPerMin = 143,77, WordPerSec = 131,95
Update = 19500, Epoch = 28, LR = 0,000430, AvgCost = 191,4792, Sent = 113, SentPerMin = 107,06, WordPerSec = 91,71
Update = 19600, Epoch = 28, LR = 0,000429, AvgCost = 205,0792, Sent = 399, SentPerMin = 101,78, WordPerSec = 93,68
Update = 19700, Epoch = 28, LR = 0,000427, AvgCost = 202,2285, Sent = 696, SentPerMin = 103,31, WordPerSec = 94,65
Update = 19800, Epoch = 28, LR = 0,000426, AvgCost = 201,5761, Sent = 995, SentPerMin = 103,52, WordPerSec = 94,94
Update = 19802, Epoch = 28, LR = 0,000426, AvgCost = 201,5238, Sent = 1000, SentPerMin = 103,49, WordPerSec = 94,98
Update = 19900, Epoch = 29, LR = 0,000425, AvgCost = 210,6803, Sent = 280, SentPerMin = 141,01, WordPerSec = 132,32
Update = 20000, Epoch = 29, LR = 0,000424, AvgCost = 201,4314, Sent = 588, SentPerMin = 142,10, WordPerSec = 129,99
Update = 20100, Epoch = 29, LR = 0,000423, AvgCost = 202,0223, Sent = 877, SentPerMin = 97,08, WordPerSec = 89,53
Update = 20140, Epoch = 29, LR = 0,000423, AvgCost = 200,7316, Sent = 1000, SentPerMin = 98,35, WordPerSec = 90,26
Update = 20200, Epoch = 30, LR = 0,000422, AvgCost = 203,1692, Sent = 173, SentPerMin = 141,49, WordPerSec = 130,18
Update = 20300, Epoch = 30, LR = 0,000421, AvgCost = 204,9951, Sent = 469, SentPerMin = 141,44, WordPerSec = 130,47
Update = 20400, Epoch = 30, LR = 0,000420, AvgCost = 203,3586, Sent = 766, SentPerMin = 135,65, WordPerSec = 125,17
Update = 20478, Epoch = 30, LR = 0,000419, AvgCost = 200,6981, Sent = 1000, SentPerMin = 138,92, WordPerSec = 127,50
Update = 20500, Epoch = 31, LR = 0,000419, AvgCost = 192,9566, Sent = 71, SentPerMin = 105,39, WordPerSec = 91,22
Update = 20600, Epoch = 31, LR = 0,000418, AvgCost = 203,8004, Sent = 358, SentPerMin = 102,30, WordPerSec = 93,97
Update = 20700, Epoch = 31, LR = 0,000417, AvgCost = 201,4355, Sent = 652, SentPerMin = 107,02, WordPerSec = 98,27
Update = 20800, Epoch = 31, LR = 0,000416, AvgCost = 199,8556, Sent = 955, SentPerMin = 106,65, WordPerSec = 97,84
Update = 20816, Epoch = 31, LR = 0,000416, AvgCost = 199,8106, Sent = 1000, SentPerMin = 106,69, WordPerSec = 97,91
Update = 20900, Epoch = 32, LR = 0,000415, AvgCost = 201,1812, Sent = 244, SentPerMin = 147,32, WordPerSec = 134,59
Update = 21000, Epoch = 32, LR = 0,000414, AvgCost = 200,7708, Sent = 542, SentPerMin = 139,99, WordPerSec = 128,58
Update = 21100, Epoch = 32, LR = 0,000413, AvgCost = 199,9954, Sent = 835, SentPerMin = 138,79, WordPerSec = 127,69
Update = 21154, Epoch = 32, LR = 0,000413, AvgCost = 198,6982, Sent = 1000, SentPerMin = 140,42, WordPerSec = 128,87
Update = 21200, Epoch = 33, LR = 0,000412, AvgCost = 197,0560, Sent = 136, SentPerMin = 106,76, WordPerSec = 95,00
Update = 21300, Epoch = 33, LR = 0,000411, AvgCost = 201,7380, Sent = 428, SentPerMin = 103,46, WordPerSec = 95,16
Update = 21400, Epoch = 33, LR = 0,000410, AvgCost = 199,1478, Sent = 725, SentPerMin = 108,37, WordPerSec = 99,34
Update = 21492, Epoch = 33, LR = 0,000409, AvgCost = 198,7385, Sent = 1000, SentPerMin = 107,57, WordPerSec = 98,73
Update = 21500, Epoch = 34, LR = 0,000409, AvgCost = 180,5036, Sent = 23, SentPerMin = 145,89, WordPerSec = 127,18
Update = 21600, Epoch = 34, LR = 0,000408, AvgCost = 204,7908, Sent = 311, SentPerMin = 143,75, WordPerSec = 133,65
Update = 21700, Epoch = 34, LR = 0,000407, AvgCost = 200,1717, Sent = 611, SentPerMin = 135,80, WordPerSec = 125,08
Update = 21800, Epoch = 34, LR = 0,000406, AvgCost = 199,1101, Sent = 906, SentPerMin = 139,10, WordPerSec = 128,32
Update = 21830, Epoch = 34, LR = 0,000406, AvgCost = 197,6962, Sent = 1000, SentPerMin = 140,49, WordPerSec = 128,93
Update = 21900, Epoch = 35, LR = 0,000405, AvgCost = 203,1262, Sent = 200, SentPerMin = 104,76, WordPerSec = 96,13
Update = 22000, Epoch = 35, LR = 0,000405, AvgCost = 199,8626, Sent = 501, SentPerMin = 104,11, WordPerSec = 95,37
Update = 22100, Epoch = 35, LR = 0,000404, AvgCost = 200,0259, Sent = 795, SentPerMin = 107,46, WordPerSec = 99,01
Update = 22168, Epoch = 35, LR = 0,000403, AvgCost = 197,8076, Sent = 1000, SentPerMin = 107,73, WordPerSec = 98,87
Update = 22200, Epoch = 36, LR = 0,000403, AvgCost = 191,6785, Sent = 99, SentPerMin = 147,52, WordPerSec = 128,55
Update = 22300, Epoch = 36, LR = 0,000402, AvgCost = 199,3412, Sent = 388, SentPerMin = 144,07, WordPerSec = 132,04
Update = 22400, Epoch = 36, LR = 0,000401, AvgCost = 197,5232, Sent = 683, SentPerMin = 138,47, WordPerSec = 126,88
Update = 22500, Epoch = 36, LR = 0,000400, AvgCost = 196,6351, Sent = 984, SentPerMin = 140,58, WordPerSec = 129,01
Update = 22506, Epoch = 36, LR = 0,000400, AvgCost = 196,5213, Sent = 1000, SentPerMin = 140,73, WordPerSec = 129,15
Update = 22600, Epoch = 37, LR = 0,000399, AvgCost = 205,6818, Sent = 269, SentPerMin = 103,90, WordPerSec = 97,12
Update = 22700, Epoch = 37, LR = 0,000398, AvgCost = 198,6277, Sent = 573, SentPerMin = 108,06, WordPerSec = 99,24
Update = 22800, Epoch = 37, LR = 0,000397, AvgCost = 197,8393, Sent = 865, SentPerMin = 107,65, WordPerSec = 99,15
Update = 22844, Epoch = 37, LR = 0,000397, AvgCost = 196,7843, Sent = 1000, SentPerMin = 107,78, WordPerSec = 98,92
Update = 22900, Epoch = 38, LR = 0,000396, AvgCost = 200,1277, Sent = 161, SentPerMin = 143,60, WordPerSec = 131,72
Update = 23000, Epoch = 38, LR = 0,000396, AvgCost = 200,5625, Sent = 456, SentPerMin = 142,89, WordPerSec = 132,02
Update = 23100, Epoch = 38, LR = 0,000395, AvgCost = 197,0692, Sent = 757, SentPerMin = 137,48, WordPerSec = 126,46
Update = 23182, Epoch = 38, LR = 0,000394, AvgCost = 195,7018, Sent = 1000, SentPerMin = 140,80, WordPerSec = 129,22
Update = 23200, Epoch = 39, LR = 0,000394, AvgCost = 184,3498, Sent = 59, SentPerMin = 113,18, WordPerSec = 94,58
Update = 23300, Epoch = 39, LR = 0,000393, AvgCost = 201,3038, Sent = 343, SentPerMin = 104,03, WordPerSec = 96,19
Update = 23400, Epoch = 39, LR = 0,000392, AvgCost = 197,2274, Sent = 640, SentPerMin = 108,35, WordPerSec = 99,57
Update = 23500, Epoch = 39, LR = 0,000391, AvgCost = 195,8198, Sent = 943, SentPerMin = 107,49, WordPerSec = 98,66
Update = 23520, Epoch = 39, LR = 0,000391, AvgCost = 195,7340, Sent = 1000, SentPerMin = 107,70, WordPerSec = 98,84
Starting inference...
Inference results:
在在在和的的的的的的的的的的和    。
和的和的的的
在的的和的         和
。
在 的 的 的     ( (                                 。



             */


            /*
             Update = 10100, Epoch = 0, LR = 0,000597, AvgCost = 265,9689, Sent = 285, SentPerMin = 629,84, WordPerSec = 591,42
Update = 10200, Epoch = 0, LR = 0,000594, AvgCost = 247,9105, Sent = 592, SentPerMin = 664,90, WordPerSec = 609,92
Update = 10300, Epoch = 0, LR = 0,000591, AvgCost = 245,4728, Sent = 882, SentPerMin = 666,61, WordPerSec = 615,83
Update = 10338, Epoch = 0, LR = 0,000590, AvgCost = 242,3184, Sent = 1000, SentPerMin = 673,25, WordPerSec = 617,89
Update = 10400, Epoch = 1, LR = 0,000588, AvgCost = 232,7651, Sent = 178, SentPerMin = 941,07, WordPerSec = 868,38
Update = 10500, Epoch = 1, LR = 0,000586, AvgCost = 233,6978, Sent = 475, SentPerMin = 959,50, WordPerSec = 883,82
Update = 10600, Epoch = 1, LR = 0,000583, AvgCost = 232,2169, Sent = 771, SentPerMin = 961,04, WordPerSec = 888,29
Update = 10676, Epoch = 1, LR = 0,000581, AvgCost = 229,0361, Sent = 1000, SentPerMin = 972,92, WordPerSec = 892,91
Update = 10700, Epoch = 2, LR = 0,000580, AvgCost = 220,8782, Sent = 76, SentPerMin = 970,34, WordPerSec = 855,86
Update = 10800, Epoch = 2, LR = 0,000577, AvgCost = 230,7474, Sent = 363, SentPerMin = 950,88, WordPerSec = 876,83
Update = 10900, Epoch = 2, LR = 0,000575, AvgCost = 227,7433, Sent = 656, SentPerMin = 956,92, WordPerSec = 881,07
Update = 11000, Epoch = 2, LR = 0,000572, AvgCost = 225,1313, Sent = 961, SentPerMin = 968,69, WordPerSec = 887,87
Update = 11014, Epoch = 2, LR = 0,000572, AvgCost = 225,5052, Sent = 1000, SentPerMin = 968,79, WordPerSec = 889,12
Update = 11100, Epoch = 3, LR = 0,000569, AvgCost = 228,1261, Sent = 248, SentPerMin = 946,80, WordPerSec = 874,65
Update = 11200, Epoch = 3, LR = 0,000567, AvgCost = 224,3253, Sent = 549, SentPerMin = 959,88, WordPerSec = 879,48
Update = 11300, Epoch = 3, LR = 0,000564, AvgCost = 224,5281, Sent = 841, SentPerMin = 960,34, WordPerSec = 884,32
Update = 11352, Epoch = 3, LR = 0,000563, AvgCost = 223,0774, Sent = 1000, SentPerMin = 967,47, WordPerSec = 887,91
Update = 11400, Epoch = 4, LR = 0,000562, AvgCost = 220,0683, Sent = 140, SentPerMin = 962,46, WordPerSec = 863,92
Update = 11500, Epoch = 4, LR = 0,000560, AvgCost = 227,7901, Sent = 431, SentPerMin = 949,33, WordPerSec = 879,98
Update = 11600, Epoch = 4, LR = 0,000557, AvgCost = 221,2700, Sent = 732, SentPerMin = 967,99, WordPerSec = 884,99
Update = 11690, Epoch = 4, LR = 0,000555, AvgCost = 221,5871, Sent = 1000, SentPerMin = 969,09, WordPerSec = 889,40
Update = 11700, Epoch = 5, LR = 0,000555, AvgCost = 184,4595, Sent = 33, SentPerMin = 1042,95, WordPerSec = 805,92
Update = 11800, Epoch = 5, LR = 0,000552, AvgCost = 228,4814, Sent = 317, SentPerMin = 942,91, WordPerSec = 878,07
Update = 11900, Epoch = 5, LR = 0,000550, AvgCost = 222,4224, Sent = 618, SentPerMin = 961,36, WordPerSec = 882,31
Update = 12000, Epoch = 5, LR = 0,000548, AvgCost = 222,1204, Sent = 914, SentPerMin = 963,11, WordPerSec = 887,50
Update = 12028, Epoch = 5, LR = 0,000547, AvgCost = 220,8821, Sent = 1000, SentPerMin = 968,84, WordPerSec = 889,17
Update = 12100, Epoch = 6, LR = 0,000545, AvgCost = 229,1927, Sent = 205, SentPerMin = 932,75, WordPerSec = 870,87
Update = 12200, Epoch = 6, LR = 0,000543, AvgCost = 221,7027, Sent = 506, SentPerMin = 954,21, WordPerSec = 875,23
Update = 12300, Epoch = 6, LR = 0,000541, AvgCost = 221,8920, Sent = 802, SentPerMin = 959,94, WordPerSec = 883,75
Update = 12366, Epoch = 6, LR = 0,000540, AvgCost = 219,7883, Sent = 1000, SentPerMin = 967,60, WordPerSec = 888,03
Update = 12400, Epoch = 7, LR = 0,000539, AvgCost = 210,3869, Sent = 106, SentPerMin = 986,06, WordPerSec = 849,31
Update = 12500, Epoch = 7, LR = 0,000537, AvgCost = 222,4834, Sent = 393, SentPerMin = 948,35, WordPerSec = 874,03
Update = 12600, Epoch = 7, LR = 0,000535, AvgCost = 219,4309, Sent = 689, SentPerMin = 963,82, WordPerSec = 883,50
Update = 12700, Epoch = 7, LR = 0,000532, AvgCost = 219,0420, Sent = 989, SentPerMin = 967,06, WordPerSec = 887,78
Update = 12704, Epoch = 7, LR = 0,000532, AvgCost = 218,8796, Sent = 1000, SentPerMin = 967,14, WordPerSec = 887,61
Update = 12800, Epoch = 8, LR = 0,000530, AvgCost = 229,0397, Sent = 275, SentPerMin = 936,63, WordPerSec = 879,24
Update = 12900, Epoch = 8, LR = 0,000528, AvgCost = 219,2791, Sent = 581, SentPerMin = 961,40, WordPerSec = 881,94
Update = 13000, Epoch = 8, LR = 0,000526, AvgCost = 219,7967, Sent = 870, SentPerMin = 960,72, WordPerSec = 887,01
Update = 13042, Epoch = 8, LR = 0,000525, AvgCost = 218,2396, Sent = 1000, SentPerMin = 968,46, WordPerSec = 888,82
Update = 13100, Epoch = 9, LR = 0,000524, AvgCost = 221,5529, Sent = 166, SentPerMin = 938,94, WordPerSec = 865,22
Update = 13200, Epoch = 9, LR = 0,000522, AvgCost = 221,6481, Sent = 463, SentPerMin = 945,90, WordPerSec = 873,07
Update = 13300, Epoch = 9, LR = 0,000520, AvgCost = 219,9088, Sent = 761, SentPerMin = 953,70, WordPerSec = 878,22
Update = 13380, Epoch = 9, LR = 0,000519, AvgCost = 217,8530, Sent = 1000, SentPerMin = 963,61, WordPerSec = 884,37
Update = 13400, Epoch = 10, LR = 0,000518, AvgCost = 215,9488, Sent = 64, SentPerMin = 714,90, WordPerSec = 623,31
Update = 13500, Epoch = 10, LR = 0,000516, AvgCost = 221,5982, Sent = 352, SentPerMin = 677,97, WordPerSec = 623,46
Update = 13600, Epoch = 10, LR = 0,000514, AvgCost = 218,7306, Sent = 645, SentPerMin = 679,36, WordPerSec = 625,32
Update = 13700, Epoch = 10, LR = 0,000513, AvgCost = 217,8274, Sent = 948, SentPerMin = 684,21, WordPerSec = 629,00
Update = 13718, Epoch = 10, LR = 0,000512, AvgCost = 217,3322, Sent = 1000, SentPerMin = 685,18, WordPerSec = 628,84
Update = 13800, Epoch = 11, LR = 0,000511, AvgCost = 219,8759, Sent = 239, SentPerMin = 965,98, WordPerSec = 882,18
Update = 13900, Epoch = 11, LR = 0,000509, AvgCost = 218,8609, Sent = 536, SentPerMin = 961,10, WordPerSec = 883,42
Update = 14000, Epoch = 11, LR = 0,000507, AvgCost = 217,3790, Sent = 831, SentPerMin = 966,36, WordPerSec = 887,34
Update = 14056, Epoch = 11, LR = 0,000506, AvgCost = 216,6796, Sent = 1000, SentPerMin = 971,37, WordPerSec = 891,49
Update = 14100, Epoch = 12, LR = 0,000505, AvgCost = 216,4633, Sent = 131, SentPerMin = 977,82, WordPerSec = 870,21
Update = 14200, Epoch = 12, LR = 0,000504, AvgCost = 218,5137, Sent = 423, SentPerMin = 960,16, WordPerSec = 882,38
Update = 14300, Epoch = 12, LR = 0,000502, AvgCost = 215,7539, Sent = 720, SentPerMin = 970,08, WordPerSec = 887,78
Update = 14394, Epoch = 12, LR = 0,000500, AvgCost = 215,8457, Sent = 1000, SentPerMin = 971,93, WordPerSec = 892,01
Update = 14400, Epoch = 13, LR = 0,000500, AvgCost = 219,4545, Sent = 16, SentPerMin = 878,95, WordPerSec = 857,89
Update = 14500, Epoch = 13, LR = 0,000498, AvgCost = 223,5815, Sent = 304, SentPerMin = 942,93, WordPerSec = 880,69
Update = 14600, Epoch = 13, LR = 0,000497, AvgCost = 216,6097, Sent = 606, SentPerMin = 960,78, WordPerSec = 883,01
Update = 14700, Epoch = 13, LR = 0,000495, AvgCost = 216,1611, Sent = 902, SentPerMin = 965,62, WordPerSec = 889,33
Update = 14732, Epoch = 13, LR = 0,000494, AvgCost = 214,9453, Sent = 1000, SentPerMin = 971,18, WordPerSec = 891,31
Update = 14800, Epoch = 14, LR = 0,000493, AvgCost = 220,4424, Sent = 195, SentPerMin = 951,57, WordPerSec = 874,63
Update = 14900, Epoch = 14, LR = 0,000492, AvgCost = 215,4251, Sent = 495, SentPerMin = 961,21, WordPerSec = 878,46
Update = 15000, Epoch = 14, LR = 0,000490, AvgCost = 216,4293, Sent = 789, SentPerMin = 959,36, WordPerSec = 884,60
Update = 15070, Epoch = 14, LR = 0,000489, AvgCost = 214,0629, Sent = 1000, SentPerMin = 969,19, WordPerSec = 889,49
Update = 15100, Epoch = 15, LR = 0,000488, AvgCost = 211,6417, Sent = 92, SentPerMin = 972,50, WordPerSec = 863,10
Update = 15200, Epoch = 15, LR = 0,000487, AvgCost = 216,8627, Sent = 381, SentPerMin = 950,65, WordPerSec = 874,79
Update = 15300, Epoch = 15, LR = 0,000485, AvgCost = 214,6438, Sent = 676, SentPerMin = 963,62, WordPerSec = 884,67
Update = 15400, Epoch = 15, LR = 0,000483, AvgCost = 212,3441, Sent = 979, SentPerMin = 972,72, WordPerSec = 890,69
Update = 15408, Epoch = 15, LR = 0,000483, AvgCost = 213,3909, Sent = 1000, SentPerMin = 970,78, WordPerSec = 890,95
Update = 15500, Epoch = 16, LR = 0,000482, AvgCost = 220,3518, Sent = 265, SentPerMin = 948,18, WordPerSec = 879,96
Update = 15600, Epoch = 16, LR = 0,000480, AvgCost = 214,5136, Sent = 566, SentPerMin = 960,31, WordPerSec = 883,85
Update = 15700, Epoch = 16, LR = 0,000479, AvgCost = 213,9089, Sent = 859, SentPerMin = 961,94, WordPerSec = 886,43
Update = 15746, Epoch = 16, LR = 0,000478, AvgCost = 212,4370, Sent = 1000, SentPerMin = 970,50, WordPerSec = 890,70
Update = 15800, Epoch = 17, LR = 0,000477, AvgCost = 214,4700, Sent = 157, SentPerMin = 957,72, WordPerSec = 871,20
Update = 15900, Epoch = 17, LR = 0,000476, AvgCost = 216,0634, Sent = 450, SentPerMin = 954,44, WordPerSec = 880,10
Update = 16000, Epoch = 17, LR = 0,000474, AvgCost = 211,9223, Sent = 752, SentPerMin = 962,91, WordPerSec = 882,82
Update = 16084, Epoch = 17, LR = 0,000473, AvgCost = 211,5446, Sent = 1000, SentPerMin = 969,61, WordPerSec = 889,88
Update = 16100, Epoch = 18, LR = 0,000473, AvgCost = 206,1960, Sent = 51, SentPerMin = 1009,88, WordPerSec = 870,60
Update = 16200, Epoch = 18, LR = 0,000471, AvgCost = 217,2652, Sent = 336, SentPerMin = 949,66, WordPerSec = 879,85
Update = 16300, Epoch = 18, LR = 0,000470, AvgCost = 211,7446, Sent = 636, SentPerMin = 962,63, WordPerSec = 881,98
Update = 16400, Epoch = 18, LR = 0,000469, AvgCost = 211,4456, Sent = 935, SentPerMin = 966,17, WordPerSec = 888,15
Update = 16422, Epoch = 18, LR = 0,000468, AvgCost = 210,8969, Sent = 1000, SentPerMin = 969,83, WordPerSec = 890,08
Update = 16500, Epoch = 19, LR = 0,000467, AvgCost = 214,8782, Sent = 225, SentPerMin = 957,47, WordPerSec = 880,17
Update = 16600, Epoch = 19, LR = 0,000466, AvgCost = 213,3544, Sent = 524, SentPerMin = 958,01, WordPerSec = 881,49
Update = 16700, Epoch = 19, LR = 0,000464, AvgCost = 211,8707, Sent = 819, SentPerMin = 960,84, WordPerSec = 883,78
Update = 16760, Epoch = 19, LR = 0,000463, AvgCost = 210,4926, Sent = 1000, SentPerMin = 967,44, WordPerSec = 887,88
Update = 16800, Epoch = 20, LR = 0,000463, AvgCost = 208,6809, Sent = 122, SentPerMin = 711,23, WordPerSec = 620,77
Update = 16900, Epoch = 20, LR = 0,000462, AvgCost = 214,2487, Sent = 411, SentPerMin = 672,85, WordPerSec = 619,70
Update = 17000, Epoch = 20, LR = 0,000460, AvgCost = 210,5569, Sent = 709, SentPerMin = 683,21, WordPerSec = 624,07
Update = 17098, Epoch = 20, LR = 0,000459, AvgCost = 210,8173, Sent = 1000, SentPerMin = 679,41, WordPerSec = 623,54
Update = 17100, Epoch = 21, LR = 0,000459, AvgCost = 190,1785, Sent = 6, SentPerMin = 923,92, WordPerSec = 839,22
Update = 17200, Epoch = 21, LR = 0,000457, AvgCost = 219,4296, Sent = 292, SentPerMin = 937,96, WordPerSec = 877,57
Update = 17300, Epoch = 21, LR = 0,000456, AvgCost = 212,3969, Sent = 595, SentPerMin = 959,04, WordPerSec = 882,69
Update = 17400, Epoch = 21, LR = 0,000455, AvgCost = 211,6511, Sent = 890, SentPerMin = 962,69, WordPerSec = 887,42
Update = 17436, Epoch = 21, LR = 0,000454, AvgCost = 210,1146, Sent = 1000, SentPerMin = 968,86, WordPerSec = 889,19
Update = 17500, Epoch = 22, LR = 0,000454, AvgCost = 216,5591, Sent = 184, SentPerMin = 945,46, WordPerSec = 868,64
Update = 17600, Epoch = 22, LR = 0,000452, AvgCost = 212,1881, Sent = 482, SentPerMin = 957,45, WordPerSec = 880,24
Update = 17700, Epoch = 22, LR = 0,000451, AvgCost = 212,6578, Sent = 775, SentPerMin = 955,17, WordPerSec = 884,98
Update = 17774, Epoch = 22, LR = 0,000450, AvgCost = 209,3013, Sent = 1000, SentPerMin = 969,06, WordPerSec = 889,37
Update = 17800, Epoch = 23, LR = 0,000450, AvgCost = 204,0007, Sent = 81, SentPerMin = 965,77, WordPerSec = 857,27
Update = 17900, Epoch = 23, LR = 0,000448, AvgCost = 213,7913, Sent = 368, SentPerMin = 944,52, WordPerSec = 874,10
Update = 18000, Epoch = 23, LR = 0,000447, AvgCost = 210,1824, Sent = 663, SentPerMin = 957,54, WordPerSec = 879,77
Update = 18100, Epoch = 23, LR = 0,000446, AvgCost = 207,7557, Sent = 969, SentPerMin = 970,54, WordPerSec = 887,49
Update = 18112, Epoch = 23, LR = 0,000446, AvgCost = 208,6239, Sent = 1000, SentPerMin = 968,03, WordPerSec = 888,42
Update = 18200, Epoch = 24, LR = 0,000445, AvgCost = 218,2449, Sent = 251, SentPerMin = 932,75, WordPerSec = 871,37
Update = 18300, Epoch = 24, LR = 0,000444, AvgCost = 208,9777, Sent = 556, SentPerMin = 958,84, WordPerSec = 877,64
Update = 18400, Epoch = 24, LR = 0,000442, AvgCost = 209,4665, Sent = 847, SentPerMin = 957,33, WordPerSec = 882,38
Update = 18450, Epoch = 24, LR = 0,000442, AvgCost = 207,9503, Sent = 1000, SentPerMin = 965,59, WordPerSec = 886,19
Update = 18500, Epoch = 25, LR = 0,000441, AvgCost = 210,3111, Sent = 145, SentPerMin = 945,51, WordPerSec = 859,98
Update = 18600, Epoch = 25, LR = 0,000440, AvgCost = 212,9551, Sent = 437, SentPerMin = 950,66, WordPerSec = 880,97
Update = 18700, Epoch = 25, LR = 0,000439, AvgCost = 207,7266, Sent = 739, SentPerMin = 964,35, WordPerSec = 884,01
Update = 18788, Epoch = 25, LR = 0,000438, AvgCost = 207,4015, Sent = 1000, SentPerMin = 969,74, WordPerSec = 890,00
Update = 18800, Epoch = 26, LR = 0,000438, AvgCost = 169,1117, Sent = 41, SentPerMin = 1072,28, WordPerSec = 825,57
Update = 18900, Epoch = 26, LR = 0,000436, AvgCost = 213,2662, Sent = 324, SentPerMin = 943,26, WordPerSec = 876,15
Update = 19000, Epoch = 26, LR = 0,000435, AvgCost = 208,1162, Sent = 624, SentPerMin = 957,53, WordPerSec = 877,81
Update = 19100, Epoch = 26, LR = 0,000434, AvgCost = 207,5637, Sent = 921, SentPerMin = 962,24, WordPerSec = 885,31
Update = 19126, Epoch = 26, LR = 0,000434, AvgCost = 206,7071, Sent = 1000, SentPerMin = 966,73, WordPerSec = 887,23
Update = 19200, Epoch = 27, LR = 0,000433, AvgCost = 211,2114, Sent = 213, SentPerMin = 945,56, WordPerSec = 872,31
Update = 19300, Epoch = 27, LR = 0,000432, AvgCost = 209,2418, Sent = 511, SentPerMin = 949,08, WordPerSec = 875,16
Update = 19400, Epoch = 27, LR = 0,000431, AvgCost = 207,8073, Sent = 808, SentPerMin = 957,20, WordPerSec = 881,70
Update = 19464, Epoch = 27, LR = 0,000430, AvgCost = 205,9207, Sent = 1000, SentPerMin = 964,62, WordPerSec = 885,30
Update = 19500, Epoch = 28, LR = 0,000430, AvgCost = 194,1481, Sent = 113, SentPerMin = 1006,84, WordPerSec = 862,50
Update = 19600, Epoch = 28, LR = 0,000429, AvgCost = 208,1119, Sent = 399, SentPerMin = 954,26, WordPerSec = 878,36
Update = 19700, Epoch = 28, LR = 0,000427, AvgCost = 205,8894, Sent = 696, SentPerMin = 965,66, WordPerSec = 884,75
Update = 19800, Epoch = 28, LR = 0,000426, AvgCost = 205,3443, Sent = 995, SentPerMin = 967,10, WordPerSec = 886,96
Update = 19802, Epoch = 28, LR = 0,000426, AvgCost = 205,2933, Sent = 1000, SentPerMin = 966,50, WordPerSec = 887,02
Update = 19900, Epoch = 29, LR = 0,000425, AvgCost = 214,6276, Sent = 280, SentPerMin = 939,41, WordPerSec = 881,48
Update = 20000, Epoch = 29, LR = 0,000424, AvgCost = 205,6823, Sent = 588, SentPerMin = 963,90, WordPerSec = 881,72
Update = 20100, Epoch = 29, LR = 0,000423, AvgCost = 206,3356, Sent = 877, SentPerMin = 677,60, WordPerSec = 624,92
Update = 20140, Epoch = 29, LR = 0,000423, AvgCost = 205,0045, Sent = 1000, SentPerMin = 707,00, WordPerSec = 648,86
Update = 20200, Epoch = 30, LR = 0,000422, AvgCost = 209,2744, Sent = 173, SentPerMin = 942,00, WordPerSec = 866,68
Update = 20300, Epoch = 30, LR = 0,000421, AvgCost = 210,9209, Sent = 469, SentPerMin = 858,21, WordPerSec = 791,64
Update = 20400, Epoch = 30, LR = 0,000420, AvgCost = 209,3008, Sent = 766, SentPerMin = 776,28, WordPerSec = 716,30
Update = 20478, Epoch = 30, LR = 0,000419, AvgCost = 206,4738, Sent = 1000, SentPerMin = 756,20, WordPerSec = 694,01
Update = 20500, Epoch = 31, LR = 0,000419, AvgCost = 198,8754, Sent = 71, SentPerMin = 1007,74, WordPerSec = 872,19
Update = 20600, Epoch = 31, LR = 0,000418, AvgCost = 209,4978, Sent = 358, SentPerMin = 949,94, WordPerSec = 872,59
Update = 20700, Epoch = 31, LR = 0,000417, AvgCost = 206,8911, Sent = 652, SentPerMin = 955,35, WordPerSec = 877,28
Update = 20800, Epoch = 31, LR = 0,000416, AvgCost = 205,2279, Sent = 955, SentPerMin = 962,95, WordPerSec = 883,44
Update = 20816, Epoch = 31, LR = 0,000416, AvgCost = 205,1705, Sent = 1000, SentPerMin = 962,76, WordPerSec = 883,59
Update = 20900, Epoch = 32, LR = 0,000415, AvgCost = 207,2457, Sent = 244, SentPerMin = 953,63, WordPerSec = 871,23
Update = 21000, Epoch = 32, LR = 0,000414, AvgCost = 206,7225, Sent = 542, SentPerMin = 925,08, WordPerSec = 849,64
Update = 21100, Epoch = 32, LR = 0,000413, AvgCost = 205,9438, Sent = 835, SentPerMin = 920,42, WordPerSec = 846,80
Update = 21154, Epoch = 32, LR = 0,000413, AvgCost = 204,5199, Sent = 1000, SentPerMin = 930,04, WordPerSec = 853,56
Update = 21200, Epoch = 33, LR = 0,000412, AvgCost = 202,4813, Sent = 136, SentPerMin = 965,04, WordPerSec = 858,72
Update = 21300, Epoch = 33, LR = 0,000411, AvgCost = 207,2071, Sent = 428, SentPerMin = 945,15, WordPerSec = 869,33
Update = 21400, Epoch = 33, LR = 0,000410, AvgCost = 204,5493, Sent = 725, SentPerMin = 957,55, WordPerSec = 877,76
Update = 21492, Epoch = 33, LR = 0,000409, AvgCost = 203,8947, Sent = 1000, SentPerMin = 962,20, WordPerSec = 883,08
Update = 21500, Epoch = 34, LR = 0,000409, AvgCost = 187,6704, Sent = 23, SentPerMin = 923,18, WordPerSec = 804,77
Update = 21600, Epoch = 34, LR = 0,000408, AvgCost = 211,0063, Sent = 311, SentPerMin = 938,04, WordPerSec = 872,14
Update = 21700, Epoch = 34, LR = 0,000407, AvgCost = 206,2678, Sent = 611, SentPerMin = 950,36, WordPerSec = 875,31
Update = 21800, Epoch = 34, LR = 0,000406, AvgCost = 205,0855, Sent = 906, SentPerMin = 953,90, WordPerSec = 879,97
Update = 21830, Epoch = 34, LR = 0,000406, AvgCost = 203,5045, Sent = 1000, SentPerMin = 960,60, WordPerSec = 881,60
Update = 21900, Epoch = 35, LR = 0,000405, AvgCost = 208,5724, Sent = 200, SentPerMin = 940,06, WordPerSec = 862,66
Update = 22000, Epoch = 35, LR = 0,000405, AvgCost = 205,1817, Sent = 501, SentPerMin = 954,78, WordPerSec = 874,71
Update = 22100, Epoch = 35, LR = 0,000404, AvgCost = 205,2494, Sent = 795, SentPerMin = 954,30, WordPerSec = 879,30
Update = 22168, Epoch = 35, LR = 0,000403, AvgCost = 202,9111, Sent = 1000, SentPerMin = 962,58, WordPerSec = 883,42
Update = 22200, Epoch = 36, LR = 0,000403, AvgCost = 198,6431, Sent = 99, SentPerMin = 983,24, WordPerSec = 856,78
Update = 22300, Epoch = 36, LR = 0,000402, AvgCost = 205,4701, Sent = 388, SentPerMin = 951,94, WordPerSec = 872,49
Update = 22400, Epoch = 36, LR = 0,000401, AvgCost = 203,3952, Sent = 683, SentPerMin = 959,13, WordPerSec = 878,85
Update = 22500, Epoch = 36, LR = 0,000400, AvgCost = 202,5056, Sent = 984, SentPerMin = 963,27, WordPerSec = 884,00
Update = 22506, Epoch = 36, LR = 0,000400, AvgCost = 202,3640, Sent = 1000, SentPerMin = 962,63, WordPerSec = 883,47
Update = 22600, Epoch = 37, LR = 0,000399, AvgCost = 211,2280, Sent = 269, SentPerMin = 941,63, WordPerSec = 880,14
Update = 22700, Epoch = 37, LR = 0,000398, AvgCost = 204,0904, Sent = 573, SentPerMin = 957,04, WordPerSec = 878,95
Update = 22800, Epoch = 37, LR = 0,000397, AvgCost = 203,2409, Sent = 865, SentPerMin = 956,49, WordPerSec = 880,97
Update = 22844, Epoch = 37, LR = 0,000397, AvgCost = 202,0508, Sent = 1000, SentPerMin = 963,41, WordPerSec = 884,19
Update = 22900, Epoch = 38, LR = 0,000396, AvgCost = 206,5509, Sent = 161, SentPerMin = 936,60, WordPerSec = 859,13
Update = 23000, Epoch = 38, LR = 0,000396, AvgCost = 206,7349, Sent = 456, SentPerMin = 946,51, WordPerSec = 874,56
Update = 23100, Epoch = 38, LR = 0,000395, AvgCost = 203,1498, Sent = 757, SentPerMin = 954,06, WordPerSec = 877,56
Update = 23182, Epoch = 38, LR = 0,000394, AvgCost = 201,6154, Sent = 1000, SentPerMin = 962,94, WordPerSec = 883,76
Update = 23200, Epoch = 39, LR = 0,000394, AvgCost = 191,8913, Sent = 59, SentPerMin = 742,15, WordPerSec = 620,14
Update = 23300, Epoch = 39, LR = 0,000393, AvgCost = 207,8024, Sent = 343, SentPerMin = 672,05, WordPerSec = 621,37
Update = 23400, Epoch = 39, LR = 0,000392, AvgCost = 203,4328, Sent = 640, SentPerMin = 778,97, WordPerSec = 715,82
Update = 23500, Epoch = 39, LR = 0,000391, AvgCost = 201,8130, Sent = 943, SentPerMin = 834,13, WordPerSec = 765,64
Update = 23520, Epoch = 39, LR = 0,000391, AvgCost = 201,6023, Sent = 1000, SentPerMin = 841,52, WordPerSec = 772,32
Update = 23600, Epoch = 40, LR = 0,000391, AvgCost = 205,0499, Sent = 231, SentPerMin = 949,29, WordPerSec = 873,54
Update = 23700, Epoch = 40, LR = 0,000390, AvgCost = 204,1521, Sent = 529, SentPerMin = 838,97, WordPerSec = 772,17
Update = 23800, Epoch = 40, LR = 0,000389, AvgCost = 202,8420, Sent = 824, SentPerMin = 774,41, WordPerSec = 712,60
Update = 23858, Epoch = 40, LR = 0,000388, AvgCost = 201,5291, Sent = 1000, SentPerMin = 757,32, WordPerSec = 695,04
Update = 23900, Epoch = 41, LR = 0,000388, AvgCost = 199,7058, Sent = 126, SentPerMin = 954,16, WordPerSec = 842,33
Update = 24000, Epoch = 41, LR = 0,000387, AvgCost = 203,3543, Sent = 418, SentPerMin = 940,12, WordPerSec = 862,67
Update = 24100, Epoch = 41, LR = 0,000386, AvgCost = 200,4562, Sent = 715, SentPerMin = 954,34, WordPerSec = 872,74
Update = 24196, Epoch = 41, LR = 0,000386, AvgCost = 200,5725, Sent = 1000, SentPerMin = 956,24, WordPerSec = 877,60
Update = 24200, Epoch = 42, LR = 0,000386, AvgCost = 183,5387, Sent = 12, SentPerMin = 957,03, WordPerSec = 861,33
Update = 24300, Epoch = 42, LR = 0,000385, AvgCost = 208,3207, Sent = 299, SentPerMin = 935,32, WordPerSec = 872,44
Update = 24400, Epoch = 42, LR = 0,000384, AvgCost = 201,7664, Sent = 601, SentPerMin = 953,80, WordPerSec = 876,19
Update = 24500, Epoch = 42, LR = 0,000383, AvgCost = 200,7601, Sent = 897, SentPerMin = 957,90, WordPerSec = 881,10
Update = 24534, Epoch = 42, LR = 0,000383, AvgCost = 199,8335, Sent = 1000, SentPerMin = 961,32, WordPerSec = 882,27
Update = 24600, Epoch = 43, LR = 0,000383, AvgCost = 206,0890, Sent = 189, SentPerMin = 935,94, WordPerSec = 862,16
Update = 24700, Epoch = 43, LR = 0,000382, AvgCost = 201,1621, Sent = 490, SentPerMin = 957,64, WordPerSec = 875,46
Update = 24800, Epoch = 43, LR = 0,000381, AvgCost = 201,5720, Sent = 784, SentPerMin = 953,93, WordPerSec = 879,16
Update = 24872, Epoch = 43, LR = 0,000380, AvgCost = 199,2563, Sent = 1000, SentPerMin = 963,82, WordPerSec = 884,56
Update = 24900, Epoch = 44, LR = 0,000380, AvgCost = 194,5375, Sent = 87, SentPerMin = 985,09, WordPerSec = 865,63
Update = 25000, Epoch = 44, LR = 0,000379, AvgCost = 202,6928, Sent = 375, SentPerMin = 945,13, WordPerSec = 871,11
Update = 25100, Epoch = 44, LR = 0,000379, AvgCost = 200,5893, Sent = 668, SentPerMin = 952,73, WordPerSec = 877,92
Update = 25200, Epoch = 44, LR = 0,000378, AvgCost = 197,7492, Sent = 974, SentPerMin = 965,38, WordPerSec = 883,39
Update = 25210, Epoch = 44, LR = 0,000378, AvgCost = 198,6675, Sent = 1000, SentPerMin = 962,96, WordPerSec = 883,77
Update = 25300, Epoch = 45, LR = 0,000377, AvgCost = 206,2813, Sent = 258, SentPerMin = 937,40, WordPerSec = 873,76
Update = 25400, Epoch = 45, LR = 0,000376, AvgCost = 199,1147, Sent = 562, SentPerMin = 958,12, WordPerSec = 878,53
Update = 25500, Epoch = 45, LR = 0,000376, AvgCost = 199,4143, Sent = 854, SentPerMin = 958,26, WordPerSec = 882,44
Update = 25548, Epoch = 45, LR = 0,000375, AvgCost = 198,1892, Sent = 1000, SentPerMin = 965,20, WordPerSec = 885,82
Update = 25600, Epoch = 46, LR = 0,000375, AvgCost = 197,8640, Sent = 152, SentPerMin = 960,06, WordPerSec = 863,74
Update = 25700, Epoch = 46, LR = 0,000374, AvgCost = 202,3139, Sent = 443, SentPerMin = 946,90, WordPerSec = 875,47
Update = 25800, Epoch = 46, LR = 0,000374, AvgCost = 198,0896, Sent = 744, SentPerMin = 958,47, WordPerSec = 880,29
Update = 25886, Epoch = 46, LR = 0,000373, AvgCost = 197,5647, Sent = 1000, SentPerMin = 966,21, WordPerSec = 886,75
Update = 25900, Epoch = 47, LR = 0,000373, AvgCost = 187,4124, Sent = 45, SentPerMin = 1021,11, WordPerSec = 836,17
Update = 26000, Epoch = 47, LR = 0,000372, AvgCost = 203,2278, Sent = 330, SentPerMin = 944,96, WordPerSec = 875,95
Update = 26100, Epoch = 47, LR = 0,000371, AvgCost = 198,5548, Sent = 630, SentPerMin = 956,11, WordPerSec = 876,82
Update = 26200, Epoch = 47, LR = 0,000371, AvgCost = 197,8782, Sent = 929, SentPerMin = 960,11, WordPerSec = 882,10
Update = 26224, Epoch = 47, LR = 0,000371, AvgCost = 197,3538, Sent = 1000, SentPerMin = 962,25, WordPerSec = 883,12
Update = 26300, Epoch = 48, LR = 0,000370, AvgCost = 201,4083, Sent = 219, SentPerMin = 942,76, WordPerSec = 868,71
Update = 26400, Epoch = 48, LR = 0,000369, AvgCost = 199,1664, Sent = 519, SentPerMin = 951,70, WordPerSec = 873,31
Update = 26500, Epoch = 48, LR = 0,000369, AvgCost = 198,2238, Sent = 814, SentPerMin = 957,22, WordPerSec = 880,45
Update = 26562, Epoch = 48, LR = 0,000368, AvgCost = 196,8217, Sent = 1000, SentPerMin = 963,53, WordPerSec = 884,30
Update = 26600, Epoch = 49, LR = 0,000368, AvgCost = 186,6346, Sent = 119, SentPerMin = 726,08, WordPerSec = 617,68
Update = 26700, Epoch = 49, LR = 0,000367, AvgCost = 200,4375, Sent = 406, SentPerMin = 702,73, WordPerSec = 645,87
Update = 26800, Epoch = 49, LR = 0,000367, AvgCost = 197,3530, Sent = 702, SentPerMin = 796,05, WordPerSec = 728,01
Update = 26900, Epoch = 49, LR = 0,000366, AvgCost = 197,1272, Sent = 1000, SentPerMin = 840,62, WordPerSec = 771,49
Update = 26900, Epoch = 49, LR = 0,000366, AvgCost = 197,1272, Sent = 1000, SentPerMin = 840,61, WordPerSec = 771,49
Update = 27000, Epoch = 50, LR = 0,000365, AvgCost = 206,5670, Sent = 285, SentPerMin = 932,32, WordPerSec = 875,45
Update = 27100, Epoch = 50, LR = 0,000364, AvgCost = 198,2279, Sent = 592, SentPerMin = 818,73, WordPerSec = 751,03
Update = 27200, Epoch = 50, LR = 0,000364, AvgCost = 199,2030, Sent = 882, SentPerMin = 761,70, WordPerSec = 703,68
Update = 27238, Epoch = 50, LR = 0,000364, AvgCost = 197,4305, Sent = 1000, SentPerMin = 756,80, WordPerSec = 694,57
Update = 27300, Epoch = 51, LR = 0,000363, AvgCost = 199,8742, Sent = 178, SentPerMin = 927,87, WordPerSec = 856,20
Update = 27400, Epoch = 51, LR = 0,000362, AvgCost = 199,6174, Sent = 475, SentPerMin = 945,82, WordPerSec = 871,22
Update = 27500, Epoch = 51, LR = 0,000362, AvgCost = 199,0993, Sent = 771, SentPerMin = 948,35, WordPerSec = 876,56
Update = 27576, Epoch = 51, LR = 0,000361, AvgCost = 196,4801, Sent = 1000, SentPerMin = 961,36, WordPerSec = 882,30
Update = 27600, Epoch = 52, LR = 0,000361, AvgCost = 192,3253, Sent = 76, SentPerMin = 982,01, WordPerSec = 866,15
Update = 27700, Epoch = 52, LR = 0,000361, AvgCost = 200,0109, Sent = 363, SentPerMin = 947,77, WordPerSec = 873,97
Update = 27800, Epoch = 52, LR = 0,000360, AvgCost = 197,8264, Sent = 656, SentPerMin = 955,79, WordPerSec = 880,02
Update = 27900, Epoch = 52, LR = 0,000359, AvgCost = 195,7351, Sent = 961, SentPerMin = 965,80, WordPerSec = 885,22
Update = 27914, Epoch = 52, LR = 0,000359, AvgCost = 196,0167, Sent = 1000, SentPerMin = 965,90, WordPerSec = 886,47
Starting inference...
Inference results:
在在在
) 的在的                                                     和
) 和的的)," 的 的和
在在在                 •和和和和 ( (  和•••和和 ( (和和和和和和 (和和和和和和和和和和和和和和和和和和
) 和。

             */


            /*
             Update = 10100, Epoch = 0, LR = 0,000597, AvgCost = 265,6180, Sent = 285, SentPerMin = 615,06, WordPerSec = 577,55
Update = 10200, Epoch = 0, LR = 0,000594, AvgCost = 247,8470, Sent = 592, SentPerMin = 647,92, WordPerSec = 594,34
Update = 10300, Epoch = 0, LR = 0,000591, AvgCost = 245,8524, Sent = 882, SentPerMin = 654,78, WordPerSec = 604,90
Update = 10338, Epoch = 0, LR = 0,000590, AvgCost = 242,7014, Sent = 1000, SentPerMin = 661,72, WordPerSec = 607,30
Update = 10400, Epoch = 1, LR = 0,000588, AvgCost = 234,6115, Sent = 178, SentPerMin = 937,33, WordPerSec = 864,92
Update = 10500, Epoch = 1, LR = 0,000586, AvgCost = 234,5548, Sent = 475, SentPerMin = 949,41, WordPerSec = 874,52
Update = 10600, Epoch = 1, LR = 0,000583, AvgCost = 232,7946, Sent = 771, SentPerMin = 951,47, WordPerSec = 879,44
Update = 10676, Epoch = 1, LR = 0,000581, AvgCost = 229,4630, Sent = 1000, SentPerMin = 963,54, WordPerSec = 884,31
Update = 10700, Epoch = 2, LR = 0,000580, AvgCost = 221,1598, Sent = 76, SentPerMin = 965,38, WordPerSec = 851,48
Update = 10800, Epoch = 2, LR = 0,000577, AvgCost = 230,6709, Sent = 363, SentPerMin = 943,04, WordPerSec = 869,61
Update = 10900, Epoch = 2, LR = 0,000575, AvgCost = 227,5505, Sent = 656, SentPerMin = 951,46, WordPerSec = 876,04
Update = 11000, Epoch = 2, LR = 0,000572, AvgCost = 225,0216, Sent = 961, SentPerMin = 961,74, WordPerSec = 881,49
Update = 11014, Epoch = 2, LR = 0,000572, AvgCost = 225,4214, Sent = 1000, SentPerMin = 961,99, WordPerSec = 882,88
Update = 11100, Epoch = 3, LR = 0,000569, AvgCost = 228,8480, Sent = 248, SentPerMin = 945,40, WordPerSec = 873,35
Update = 11200, Epoch = 3, LR = 0,000567, AvgCost = 224,8003, Sent = 549, SentPerMin = 956,31, WordPerSec = 876,21
Update = 11300, Epoch = 3, LR = 0,000564, AvgCost = 224,7456, Sent = 841, SentPerMin = 955,68, WordPerSec = 880,04
Update = 11352, Epoch = 3, LR = 0,000563, AvgCost = 223,2799, Sent = 1000, SentPerMin = 963,92, WordPerSec = 884,66
Update = 11400, Epoch = 4, LR = 0,000562, AvgCost = 222,5244, Sent = 140, SentPerMin = 956,29, WordPerSec = 858,38
Update = 11500, Epoch = 4, LR = 0,000560, AvgCost = 228,7103, Sent = 431, SentPerMin = 940,70, WordPerSec = 871,99
Update = 11600, Epoch = 4, LR = 0,000557, AvgCost = 221,5627, Sent = 732, SentPerMin = 958,08, WordPerSec = 875,93
Update = 11690, Epoch = 4, LR = 0,000555, AvgCost = 221,8105, Sent = 1000, SentPerMin = 961,52, WordPerSec = 882,45
Update = 11700, Epoch = 5, LR = 0,000555, AvgCost = 184,5222, Sent = 33, SentPerMin = 1046,32, WordPerSec = 808,52
Update = 11800, Epoch = 5, LR = 0,000552, AvgCost = 228,2314, Sent = 317, SentPerMin = 937,92, WordPerSec = 873,42
Update = 11900, Epoch = 5, LR = 0,000550, AvgCost = 222,1811, Sent = 618, SentPerMin = 956,05, WordPerSec = 877,44
Update = 12000, Epoch = 5, LR = 0,000548, AvgCost = 221,5915, Sent = 914, SentPerMin = 958,24, WordPerSec = 883,01
Update = 12028, Epoch = 5, LR = 0,000547, AvgCost = 220,4772, Sent = 1000, SentPerMin = 963,52, WordPerSec = 884,28
Update = 12100, Epoch = 6, LR = 0,000545, AvgCost = 228,0454, Sent = 205, SentPerMin = 931,30, WordPerSec = 869,52
Update = 12200, Epoch = 6, LR = 0,000543, AvgCost = 221,3103, Sent = 506, SentPerMin = 953,75, WordPerSec = 874,80
Update = 12300, Epoch = 6, LR = 0,000541, AvgCost = 221,3908, Sent = 802, SentPerMin = 956,96, WordPerSec = 881,01
Update = 12366, Epoch = 6, LR = 0,000540, AvgCost = 219,5513, Sent = 1000, SentPerMin = 963,01, WordPerSec = 883,82
Update = 12400, Epoch = 7, LR = 0,000539, AvgCost = 208,7981, Sent = 106, SentPerMin = 996,25, WordPerSec = 858,09
Update = 12500, Epoch = 7, LR = 0,000537, AvgCost = 223,1160, Sent = 393, SentPerMin = 947,42, WordPerSec = 873,17
Update = 12600, Epoch = 7, LR = 0,000535, AvgCost = 219,7304, Sent = 689, SentPerMin = 963,08, WordPerSec = 882,82
Update = 12700, Epoch = 7, LR = 0,000532, AvgCost = 219,3402, Sent = 989, SentPerMin = 965,73, WordPerSec = 886,56
Update = 12704, Epoch = 7, LR = 0,000532, AvgCost = 219,1965, Sent = 1000, SentPerMin = 965,71, WordPerSec = 886,30
Update = 12800, Epoch = 8, LR = 0,000530, AvgCost = 229,9708, Sent = 275, SentPerMin = 934,06, WordPerSec = 876,83
Update = 12900, Epoch = 8, LR = 0,000528, AvgCost = 219,8980, Sent = 581, SentPerMin = 956,78, WordPerSec = 877,71
Update = 13000, Epoch = 8, LR = 0,000526, AvgCost = 220,3798, Sent = 870, SentPerMin = 956,01, WordPerSec = 882,66
Update = 13042, Epoch = 8, LR = 0,000525, AvgCost = 218,9054, Sent = 1000, SentPerMin = 964,72, WordPerSec = 885,38
Update = 13100, Epoch = 9, LR = 0,000524, AvgCost = 221,7362, Sent = 166, SentPerMin = 930,78, WordPerSec = 857,70
Update = 13200, Epoch = 9, LR = 0,000522, AvgCost = 222,4501, Sent = 463, SentPerMin = 945,55, WordPerSec = 872,75
Update = 13300, Epoch = 9, LR = 0,000520, AvgCost = 220,1850, Sent = 761, SentPerMin = 951,05, WordPerSec = 875,77
Update = 13380, Epoch = 9, LR = 0,000519, AvgCost = 218,2647, Sent = 1000, SentPerMin = 962,07, WordPerSec = 882,95
Update = 13400, Epoch = 10, LR = 0,000518, AvgCost = 213,5388, Sent = 64, SentPerMin = 718,71, WordPerSec = 626,63
Update = 13500, Epoch = 10, LR = 0,000516, AvgCost = 221,6022, Sent = 352, SentPerMin = 671,00, WordPerSec = 617,06
Update = 13600, Epoch = 10, LR = 0,000514, AvgCost = 218,8450, Sent = 645, SentPerMin = 672,38, WordPerSec = 618,91
Update = 13700, Epoch = 10, LR = 0,000513, AvgCost = 217,9728, Sent = 948, SentPerMin = 677,44, WordPerSec = 622,78
Update = 13718, Epoch = 10, LR = 0,000512, AvgCost = 217,5381, Sent = 1000, SentPerMin = 677,98, WordPerSec = 622,23
Update = 13800, Epoch = 11, LR = 0,000511, AvgCost = 219,4172, Sent = 239, SentPerMin = 962,83, WordPerSec = 879,30
Update = 13900, Epoch = 11, LR = 0,000509, AvgCost = 219,2700, Sent = 536, SentPerMin = 955,81, WordPerSec = 878,57
Update = 14000, Epoch = 11, LR = 0,000507, AvgCost = 217,5955, Sent = 831, SentPerMin = 961,65, WordPerSec = 883,02
Update = 14056, Epoch = 11, LR = 0,000506, AvgCost = 216,8910, Sent = 1000, SentPerMin = 965,76, WordPerSec = 886,34
Update = 14100, Epoch = 12, LR = 0,000505, AvgCost = 215,0144, Sent = 131, SentPerMin = 967,74, WordPerSec = 861,24
Update = 14200, Epoch = 12, LR = 0,000504, AvgCost = 218,9013, Sent = 423, SentPerMin = 952,24, WordPerSec = 875,10
Update = 14300, Epoch = 12, LR = 0,000502, AvgCost = 216,0046, Sent = 720, SentPerMin = 962,50, WordPerSec = 880,85
Update = 14394, Epoch = 12, LR = 0,000500, AvgCost = 216,2762, Sent = 1000, SentPerMin = 966,33, WordPerSec = 886,86
Update = 14400, Epoch = 13, LR = 0,000500, AvgCost = 217,7292, Sent = 16, SentPerMin = 865,92, WordPerSec = 845,18
Update = 14500, Epoch = 13, LR = 0,000498, AvgCost = 223,8882, Sent = 304, SentPerMin = 946,27, WordPerSec = 883,81
Update = 14600, Epoch = 13, LR = 0,000497, AvgCost = 217,3552, Sent = 606, SentPerMin = 958,81, WordPerSec = 881,20
Update = 14700, Epoch = 13, LR = 0,000495, AvgCost = 216,9619, Sent = 902, SentPerMin = 960,28, WordPerSec = 884,41
Update = 14732, Epoch = 13, LR = 0,000494, AvgCost = 215,7914, Sent = 1000, SentPerMin = 965,88, WordPerSec = 886,45
Update = 14800, Epoch = 14, LR = 0,000493, AvgCost = 220,8699, Sent = 195, SentPerMin = 950,09, WordPerSec = 873,27
Update = 14900, Epoch = 14, LR = 0,000492, AvgCost = 216,4026, Sent = 495, SentPerMin = 962,95, WordPerSec = 880,05
Update = 15000, Epoch = 14, LR = 0,000490, AvgCost = 217,4535, Sent = 789, SentPerMin = 958,00, WordPerSec = 883,34
Update = 15070, Epoch = 14, LR = 0,000489, AvgCost = 215,1649, Sent = 1000, SentPerMin = 967,71, WordPerSec = 888,13
Update = 15100, Epoch = 15, LR = 0,000488, AvgCost = 214,0051, Sent = 92, SentPerMin = 969,48, WordPerSec = 860,41
Update = 15200, Epoch = 15, LR = 0,000487, AvgCost = 218,0411, Sent = 381, SentPerMin = 950,77, WordPerSec = 874,91
Update = 15300, Epoch = 15, LR = 0,000485, AvgCost = 215,7413, Sent = 676, SentPerMin = 959,15, WordPerSec = 880,57
Update = 15400, Epoch = 15, LR = 0,000483, AvgCost = 213,5706, Sent = 979, SentPerMin = 968,81, WordPerSec = 887,10
Update = 15408, Epoch = 15, LR = 0,000483, AvgCost = 214,6299, Sent = 1000, SentPerMin = 966,39, WordPerSec = 886,92
Update = 15500, Epoch = 16, LR = 0,000482, AvgCost = 220,8819, Sent = 265, SentPerMin = 948,02, WordPerSec = 879,81
Update = 15600, Epoch = 16, LR = 0,000480, AvgCost = 215,6976, Sent = 566, SentPerMin = 957,36, WordPerSec = 881,13
Update = 15700, Epoch = 16, LR = 0,000479, AvgCost = 215,0446, Sent = 859, SentPerMin = 958,83, WordPerSec = 883,56
Update = 15746, Epoch = 16, LR = 0,000478, AvgCost = 213,7936, Sent = 1000, SentPerMin = 965,95, WordPerSec = 886,52
Update = 15800, Epoch = 17, LR = 0,000477, AvgCost = 215,6239, Sent = 157, SentPerMin = 951,29, WordPerSec = 865,35
Update = 15900, Epoch = 17, LR = 0,000476, AvgCost = 217,2900, Sent = 450, SentPerMin = 949,70, WordPerSec = 875,73
Update = 16000, Epoch = 17, LR = 0,000474, AvgCost = 212,8760, Sent = 752, SentPerMin = 958,64, WordPerSec = 878,90
Update = 16084, Epoch = 17, LR = 0,000473, AvgCost = 212,8855, Sent = 1000, SentPerMin = 965,09, WordPerSec = 885,73
Update = 16100, Epoch = 18, LR = 0,000473, AvgCost = 206,1781, Sent = 51, SentPerMin = 999,78, WordPerSec = 861,90
Update = 16200, Epoch = 18, LR = 0,000471, AvgCost = 218,0301, Sent = 336, SentPerMin = 938,10, WordPerSec = 869,14
Update = 16300, Epoch = 18, LR = 0,000470, AvgCost = 212,7052, Sent = 636, SentPerMin = 954,30, WordPerSec = 874,35
Update = 16400, Epoch = 18, LR = 0,000469, AvgCost = 212,5451, Sent = 935, SentPerMin = 958,04, WordPerSec = 880,68
Update = 16422, Epoch = 18, LR = 0,000468, AvgCost = 212,1183, Sent = 1000, SentPerMin = 962,10, WordPerSec = 882,99
Update = 16500, Epoch = 19, LR = 0,000467, AvgCost = 214,9719, Sent = 225, SentPerMin = 949,36, WordPerSec = 872,71
Update = 16600, Epoch = 19, LR = 0,000466, AvgCost = 214,2904, Sent = 524, SentPerMin = 951,90, WordPerSec = 875,88
Update = 16700, Epoch = 19, LR = 0,000464, AvgCost = 212,6571, Sent = 819, SentPerMin = 955,20, WordPerSec = 878,59
Update = 16760, Epoch = 19, LR = 0,000463, AvgCost = 211,5788, Sent = 1000, SentPerMin = 961,28, WordPerSec = 882,23
Update = 16800, Epoch = 20, LR = 0,000463, AvgCost = 208,3749, Sent = 122, SentPerMin = 701,70, WordPerSec = 612,46
Update = 16900, Epoch = 20, LR = 0,000462, AvgCost = 214,7132, Sent = 411, SentPerMin = 667,06, WordPerSec = 614,37
Update = 17000, Epoch = 20, LR = 0,000460, AvgCost = 211,0654, Sent = 709, SentPerMin = 678,00, WordPerSec = 619,32
Update = 17098, Epoch = 20, LR = 0,000459, AvgCost = 211,8208, Sent = 1000, SentPerMin = 676,47, WordPerSec = 620,84
Update = 17100, Epoch = 21, LR = 0,000459, AvgCost = 189,7209, Sent = 6, SentPerMin = 903,16, WordPerSec = 820,37
Update = 17200, Epoch = 21, LR = 0,000457, AvgCost = 219,5767, Sent = 292, SentPerMin = 935,68, WordPerSec = 875,44
Update = 17300, Epoch = 21, LR = 0,000456, AvgCost = 213,0207, Sent = 595, SentPerMin = 950,02, WordPerSec = 874,40
Update = 17400, Epoch = 21, LR = 0,000455, AvgCost = 212,3566, Sent = 890, SentPerMin = 955,21, WordPerSec = 880,52
Update = 17436, Epoch = 21, LR = 0,000454, AvgCost = 211,0504, Sent = 1000, SentPerMin = 961,08, WordPerSec = 882,04
Update = 17500, Epoch = 22, LR = 0,000454, AvgCost = 216,6599, Sent = 184, SentPerMin = 946,36, WordPerSec = 869,47
Update = 17600, Epoch = 22, LR = 0,000452, AvgCost = 213,0579, Sent = 482, SentPerMin = 955,08, WordPerSec = 878,07
Update = 17700, Epoch = 22, LR = 0,000451, AvgCost = 213,5116, Sent = 775, SentPerMin = 950,59, WordPerSec = 880,74
Update = 17774, Epoch = 22, LR = 0,000450, AvgCost = 210,4795, Sent = 1000, SentPerMin = 963,93, WordPerSec = 884,66
Update = 17800, Epoch = 23, LR = 0,000450, AvgCost = 204,9914, Sent = 81, SentPerMin = 967,22, WordPerSec = 858,56
Update = 17900, Epoch = 23, LR = 0,000448, AvgCost = 214,6533, Sent = 368, SentPerMin = 944,05, WordPerSec = 873,68
Update = 18000, Epoch = 23, LR = 0,000447, AvgCost = 211,0023, Sent = 663, SentPerMin = 957,40, WordPerSec = 879,64
Update = 18100, Epoch = 23, LR = 0,000446, AvgCost = 208,8947, Sent = 969, SentPerMin = 965,46, WordPerSec = 882,85
Update = 18112, Epoch = 23, LR = 0,000446, AvgCost = 209,8684, Sent = 1000, SentPerMin = 962,81, WordPerSec = 883,63
Update = 18200, Epoch = 24, LR = 0,000445, AvgCost = 218,4605, Sent = 251, SentPerMin = 936,78, WordPerSec = 875,14
Update = 18300, Epoch = 24, LR = 0,000444, AvgCost = 209,8545, Sent = 556, SentPerMin = 961,81, WordPerSec = 880,36
Update = 18400, Epoch = 24, LR = 0,000442, AvgCost = 210,4450, Sent = 847, SentPerMin = 957,49, WordPerSec = 882,52
Update = 18450, Epoch = 24, LR = 0,000442, AvgCost = 209,2120, Sent = 1000, SentPerMin = 964,77, WordPerSec = 885,43
Update = 18500, Epoch = 25, LR = 0,000441, AvgCost = 211,3540, Sent = 145, SentPerMin = 945,00, WordPerSec = 859,51
Update = 18600, Epoch = 25, LR = 0,000440, AvgCost = 214,2757, Sent = 437, SentPerMin = 946,95, WordPerSec = 877,53
Update = 18700, Epoch = 25, LR = 0,000439, AvgCost = 208,5904, Sent = 739, SentPerMin = 959,20, WordPerSec = 879,29
Update = 18788, Epoch = 25, LR = 0,000438, AvgCost = 208,7662, Sent = 1000, SentPerMin = 963,98, WordPerSec = 884,71
Update = 18800, Epoch = 26, LR = 0,000438, AvgCost = 171,0614, Sent = 41, SentPerMin = 1081,54, WordPerSec = 832,70
Update = 18900, Epoch = 26, LR = 0,000436, AvgCost = 214,4596, Sent = 324, SentPerMin = 939,66, WordPerSec = 872,81
Update = 19000, Epoch = 26, LR = 0,000435, AvgCost = 209,3750, Sent = 624, SentPerMin = 953,02, WordPerSec = 873,68
Update = 19100, Epoch = 26, LR = 0,000434, AvgCost = 208,8511, Sent = 921, SentPerMin = 958,17, WordPerSec = 881,57
Update = 19126, Epoch = 26, LR = 0,000434, AvgCost = 208,1299, Sent = 1000, SentPerMin = 961,85, WordPerSec = 882,76
Update = 19200, Epoch = 27, LR = 0,000433, AvgCost = 212,5425, Sent = 213, SentPerMin = 940,53, WordPerSec = 867,67
Update = 19300, Epoch = 27, LR = 0,000432, AvgCost = 210,9536, Sent = 511, SentPerMin = 945,58, WordPerSec = 871,93
Update = 19400, Epoch = 27, LR = 0,000431, AvgCost = 209,2084, Sent = 808, SentPerMin = 951,51, WordPerSec = 876,46
Update = 19464, Epoch = 27, LR = 0,000430, AvgCost = 207,4775, Sent = 1000, SentPerMin = 959,15, WordPerSec = 880,28
Update = 19500, Epoch = 28, LR = 0,000430, AvgCost = 196,8523, Sent = 113, SentPerMin = 1005,41, WordPerSec = 861,27
Update = 19600, Epoch = 28, LR = 0,000429, AvgCost = 210,3210, Sent = 399, SentPerMin = 943,09, WordPerSec = 868,09
Update = 19700, Epoch = 28, LR = 0,000427, AvgCost = 207,5125, Sent = 696, SentPerMin = 957,53, WordPerSec = 877,30
Update = 19800, Epoch = 28, LR = 0,000426, AvgCost = 207,0880, Sent = 995, SentPerMin = 961,81, WordPerSec = 882,11
Update = 19802, Epoch = 28, LR = 0,000426, AvgCost = 207,0370, Sent = 1000, SentPerMin = 961,19, WordPerSec = 882,15
Update = 19900, Epoch = 29, LR = 0,000425, AvgCost = 216,5991, Sent = 280, SentPerMin = 929,80, WordPerSec = 872,46
Update = 20000, Epoch = 29, LR = 0,000424, AvgCost = 207,4482, Sent = 588, SentPerMin = 954,66, WordPerSec = 873,26
Update = 20100, Epoch = 29, LR = 0,000423, AvgCost = 207,7002, Sent = 877, SentPerMin = 658,26, WordPerSec = 607,08
Update = 20140, Epoch = 29, LR = 0,000423, AvgCost = 206,4668, Sent = 1000, SentPerMin = 687,80, WordPerSec = 631,24
Update = 20200, Epoch = 30, LR = 0,000422, AvgCost = 209,4984, Sent = 173, SentPerMin = 939,95, WordPerSec = 864,79
Update = 20300, Epoch = 30, LR = 0,000421, AvgCost = 211,7697, Sent = 469, SentPerMin = 894,89, WordPerSec = 825,46
Update = 20400, Epoch = 30, LR = 0,000420, AvgCost = 210,1310, Sent = 766, SentPerMin = 789,41, WordPerSec = 728,42
Update = 20478, Epoch = 30, LR = 0,000419, AvgCost = 207,6586, Sent = 1000, SentPerMin = 766,31, WordPerSec = 703,30
Update = 20500, Epoch = 31, LR = 0,000419, AvgCost = 200,6307, Sent = 71, SentPerMin = 997,07, WordPerSec = 862,96
Update = 20600, Epoch = 31, LR = 0,000418, AvgCost = 210,5327, Sent = 358, SentPerMin = 942,12, WordPerSec = 865,40
Update = 20700, Epoch = 31, LR = 0,000417, AvgCost = 208,3971, Sent = 652, SentPerMin = 948,98, WordPerSec = 871,43
Update = 20800, Epoch = 31, LR = 0,000416, AvgCost = 206,7223, Sent = 955, SentPerMin = 957,41, WordPerSec = 878,36
Update = 20816, Epoch = 31, LR = 0,000416, AvgCost = 206,7541, Sent = 1000, SentPerMin = 957,82, WordPerSec = 879,05
Update = 20900, Epoch = 32, LR = 0,000415, AvgCost = 208,2412, Sent = 244, SentPerMin = 950,04, WordPerSec = 867,95
Update = 21000, Epoch = 32, LR = 0,000414, AvgCost = 208,3641, Sent = 542, SentPerMin = 947,40, WordPerSec = 870,14
Update = 21100, Epoch = 32, LR = 0,000413, AvgCost = 207,2885, Sent = 835, SentPerMin = 951,95, WordPerSec = 875,81
Update = 21154, Epoch = 32, LR = 0,000413, AvgCost = 205,9706, Sent = 1000, SentPerMin = 958,23, WordPerSec = 879,43
Update = 21200, Epoch = 33, LR = 0,000412, AvgCost = 203,7743, Sent = 136, SentPerMin = 963,45, WordPerSec = 857,30
Update = 21300, Epoch = 33, LR = 0,000411, AvgCost = 208,4169, Sent = 428, SentPerMin = 945,66, WordPerSec = 869,80
Update = 21400, Epoch = 33, LR = 0,000410, AvgCost = 205,6090, Sent = 725, SentPerMin = 949,55, WordPerSec = 870,42
Update = 21492, Epoch = 33, LR = 0,000409, AvgCost = 205,1691, Sent = 1000, SentPerMin = 952,57, WordPerSec = 874,24
Update = 21500, Epoch = 34, LR = 0,000409, AvgCost = 189,1688, Sent = 23, SentPerMin = 930,26, WordPerSec = 810,94
Update = 21600, Epoch = 34, LR = 0,000408, AvgCost = 211,3844, Sent = 311, SentPerMin = 941,92, WordPerSec = 875,74
Update = 21700, Epoch = 34, LR = 0,000407, AvgCost = 206,9611, Sent = 611, SentPerMin = 948,82, WordPerSec = 873,89
Update = 21800, Epoch = 34, LR = 0,000406, AvgCost = 205,7977, Sent = 906, SentPerMin = 953,27, WordPerSec = 879,39
Update = 21830, Epoch = 34, LR = 0,000406, AvgCost = 204,3681, Sent = 1000, SentPerMin = 960,51, WordPerSec = 881,52
Update = 21900, Epoch = 35, LR = 0,000405, AvgCost = 209,0291, Sent = 200, SentPerMin = 938,65, WordPerSec = 861,37
Update = 22000, Epoch = 35, LR = 0,000405, AvgCost = 206,3969, Sent = 501, SentPerMin = 950,44, WordPerSec = 870,73
Update = 22100, Epoch = 35, LR = 0,000404, AvgCost = 206,0927, Sent = 795, SentPerMin = 950,15, WordPerSec = 875,47
Update = 22168, Epoch = 35, LR = 0,000403, AvgCost = 203,9156, Sent = 1000, SentPerMin = 958,29, WordPerSec = 879,48
Update = 22200, Epoch = 36, LR = 0,000403, AvgCost = 199,1100, Sent = 99, SentPerMin = 979,60, WordPerSec = 853,60
Update = 22300, Epoch = 36, LR = 0,000402, AvgCost = 206,5099, Sent = 388, SentPerMin = 946,91, WordPerSec = 867,88
Update = 22400, Epoch = 36, LR = 0,000401, AvgCost = 204,3077, Sent = 683, SentPerMin = 952,76, WordPerSec = 873,02
Update = 22500, Epoch = 36, LR = 0,000400, AvgCost = 203,5495, Sent = 984, SentPerMin = 957,93, WordPerSec = 879,09
Update = 22506, Epoch = 36, LR = 0,000400, AvgCost = 203,4654, Sent = 1000, SentPerMin = 957,62, WordPerSec = 878,87
Update = 22600, Epoch = 37, LR = 0,000399, AvgCost = 211,7756, Sent = 269, SentPerMin = 931,57, WordPerSec = 870,74
Update = 22700, Epoch = 37, LR = 0,000398, AvgCost = 205,0850, Sent = 573, SentPerMin = 950,51, WordPerSec = 872,96
Update = 22800, Epoch = 37, LR = 0,000397, AvgCost = 203,9819, Sent = 865, SentPerMin = 951,85, WordPerSec = 876,69
Update = 22844, Epoch = 37, LR = 0,000397, AvgCost = 203,0493, Sent = 1000, SentPerMin = 958,49, WordPerSec = 879,67
Update = 22900, Epoch = 38, LR = 0,000396, AvgCost = 207,9269, Sent = 161, SentPerMin = 929,56, WordPerSec = 852,68
Update = 23000, Epoch = 38, LR = 0,000396, AvgCost = 208,1155, Sent = 456, SentPerMin = 939,69, WordPerSec = 868,25
Update = 23100, Epoch = 38, LR = 0,000395, AvgCost = 203,9875, Sent = 757, SentPerMin = 950,09, WordPerSec = 873,91
Update = 23182, Epoch = 38, LR = 0,000394, AvgCost = 202,7207, Sent = 1000, SentPerMin = 959,40, WordPerSec = 880,51
Update = 23200, Epoch = 39, LR = 0,000394, AvgCost = 194,4936, Sent = 59, SentPerMin = 740,41, WordPerSec = 618,68
Update = 23300, Epoch = 39, LR = 0,000393, AvgCost = 208,6858, Sent = 343, SentPerMin = 665,87, WordPerSec = 615,66
Update = 23400, Epoch = 39, LR = 0,000392, AvgCost = 204,0833, Sent = 640, SentPerMin = 746,97, WordPerSec = 686,42
Update = 23500, Epoch = 39, LR = 0,000391, AvgCost = 202,3849, Sent = 943, SentPerMin = 803,52, WordPerSec = 737,54
Update = 23520, Epoch = 39, LR = 0,000391, AvgCost = 202,3792, Sent = 1000, SentPerMin = 811,14, WordPerSec = 744,44
Update = 23600, Epoch = 40, LR = 0,000391, AvgCost = 205,6790, Sent = 231, SentPerMin = 923,92, WordPerSec = 850,19
Update = 23700, Epoch = 40, LR = 0,000390, AvgCost = 204,7994, Sent = 529, SentPerMin = 851,41, WordPerSec = 783,62
Update = 23800, Epoch = 40, LR = 0,000389, AvgCost = 203,3477, Sent = 824, SentPerMin = 780,92, WordPerSec = 718,59
Update = 23858, Epoch = 40, LR = 0,000388, AvgCost = 202,3833, Sent = 1000, SentPerMin = 764,74, WordPerSec = 701,85
Update = 23900, Epoch = 41, LR = 0,000388, AvgCost = 200,8317, Sent = 126, SentPerMin = 972,20, WordPerSec = 858,26
Update = 24000, Epoch = 41, LR = 0,000387, AvgCost = 204,1019, Sent = 418, SentPerMin = 946,14, WordPerSec = 868,20
Update = 24100, Epoch = 41, LR = 0,000386, AvgCost = 201,0934, Sent = 715, SentPerMin = 959,66, WordPerSec = 877,61
Update = 24196, Epoch = 41, LR = 0,000386, AvgCost = 201,5944, Sent = 1000, SentPerMin = 958,79, WordPerSec = 879,95
Update = 24200, Epoch = 42, LR = 0,000386, AvgCost = 184,3856, Sent = 12, SentPerMin = 941,38, WordPerSec = 847,24
Update = 24300, Epoch = 42, LR = 0,000385, AvgCost = 209,2475, Sent = 299, SentPerMin = 931,60, WordPerSec = 868,98
Update = 24400, Epoch = 42, LR = 0,000384, AvgCost = 202,5036, Sent = 601, SentPerMin = 945,51, WordPerSec = 868,58
Update = 24500, Epoch = 42, LR = 0,000383, AvgCost = 201,4459, Sent = 897, SentPerMin = 951,74, WordPerSec = 875,43
Update = 24534, Epoch = 42, LR = 0,000383, AvgCost = 200,7956, Sent = 1000, SentPerMin = 955,62, WordPerSec = 877,04
Update = 24600, Epoch = 43, LR = 0,000383, AvgCost = 207,4139, Sent = 189, SentPerMin = 925,52, WordPerSec = 852,55
Update = 24700, Epoch = 43, LR = 0,000382, AvgCost = 202,2414, Sent = 490, SentPerMin = 945,42, WordPerSec = 864,29
Update = 24800, Epoch = 43, LR = 0,000381, AvgCost = 202,3778, Sent = 784, SentPerMin = 945,35, WordPerSec = 871,25
Update = 24872, Epoch = 43, LR = 0,000380, AvgCost = 200,2399, Sent = 1000, SentPerMin = 955,99, WordPerSec = 877,38
Update = 24900, Epoch = 44, LR = 0,000380, AvgCost = 195,7825, Sent = 87, SentPerMin = 982,44, WordPerSec = 863,31
Update = 25000, Epoch = 44, LR = 0,000379, AvgCost = 203,9741, Sent = 375, SentPerMin = 944,04, WordPerSec = 870,11
Update = 25100, Epoch = 44, LR = 0,000379, AvgCost = 201,5805, Sent = 668, SentPerMin = 949,33, WordPerSec = 874,79
Update = 25200, Epoch = 44, LR = 0,000378, AvgCost = 198,7447, Sent = 974, SentPerMin = 961,94, WordPerSec = 880,25
Update = 25210, Epoch = 44, LR = 0,000378, AvgCost = 199,8015, Sent = 1000, SentPerMin = 959,58, WordPerSec = 880,67
Update = 25300, Epoch = 45, LR = 0,000377, AvgCost = 208,0677, Sent = 258, SentPerMin = 920,95, WordPerSec = 858,43
Update = 25400, Epoch = 45, LR = 0,000376, AvgCost = 200,3824, Sent = 562, SentPerMin = 948,55, WordPerSec = 869,75
Update = 25500, Epoch = 45, LR = 0,000376, AvgCost = 200,4021, Sent = 854, SentPerMin = 947,97, WordPerSec = 872,97
Update = 25548, Epoch = 45, LR = 0,000375, AvgCost = 199,3379, Sent = 1000, SentPerMin = 954,92, WordPerSec = 876,40
Update = 25600, Epoch = 46, LR = 0,000375, AvgCost = 199,3308, Sent = 152, SentPerMin = 944,95, WordPerSec = 850,14
Update = 25700, Epoch = 46, LR = 0,000374, AvgCost = 203,9518, Sent = 443, SentPerMin = 936,56, WordPerSec = 865,91
Update = 25800, Epoch = 46, LR = 0,000374, AvgCost = 199,3528, Sent = 744, SentPerMin = 948,16, WordPerSec = 870,83
Update = 25886, Epoch = 46, LR = 0,000373, AvgCost = 198,9170, Sent = 1000, SentPerMin = 954,84, WordPerSec = 876,32
Update = 25900, Epoch = 47, LR = 0,000373, AvgCost = 188,4860, Sent = 45, SentPerMin = 1009,85, WordPerSec = 826,95
Update = 26000, Epoch = 47, LR = 0,000372, AvgCost = 204,8800, Sent = 330, SentPerMin = 939,43, WordPerSec = 870,82
Update = 26100, Epoch = 47, LR = 0,000371, AvgCost = 199,8395, Sent = 630, SentPerMin = 949,10, WordPerSec = 870,38
Update = 26200, Epoch = 47, LR = 0,000371, AvgCost = 198,9097, Sent = 929, SentPerMin = 955,53, WordPerSec = 877,89
Update = 26224, Epoch = 47, LR = 0,000371, AvgCost = 198,6183, Sent = 1000, SentPerMin = 958,99, WordPerSec = 880,13
Update = 26300, Epoch = 48, LR = 0,000370, AvgCost = 203,3441, Sent = 219, SentPerMin = 932,34, WordPerSec = 859,12
Update = 26400, Epoch = 48, LR = 0,000369, AvgCost = 200,9122, Sent = 519, SentPerMin = 944,20, WordPerSec = 866,43
Update = 26500, Epoch = 48, LR = 0,000369, AvgCost = 199,5391, Sent = 814, SentPerMin = 948,67, WordPerSec = 872,59
Update = 26562, Epoch = 48, LR = 0,000368, AvgCost = 198,2568, Sent = 1000, SentPerMin = 955,10, WordPerSec = 876,56
Update = 26600, Epoch = 49, LR = 0,000368, AvgCost = 188,2651, Sent = 119, SentPerMin = 706,48, WordPerSec = 601,01
Update = 26700, Epoch = 49, LR = 0,000367, AvgCost = 202,1515, Sent = 406, SentPerMin = 666,27, WordPerSec = 612,36
Update = 26800, Epoch = 49, LR = 0,000367, AvgCost = 198,8275, Sent = 702, SentPerMin = 761,58, WordPerSec = 696,49
Update = 26900, Epoch = 49, LR = 0,000366, AvgCost = 198,4938, Sent = 1000, SentPerMin = 808,24, WordPerSec = 741,77
Update = 26900, Epoch = 49, LR = 0,000366, AvgCost = 198,4938, Sent = 1000, SentPerMin = 808,23, WordPerSec = 741,77
Update = 27000, Epoch = 50, LR = 0,000365, AvgCost = 208,5175, Sent = 285, SentPerMin = 908,66, WordPerSec = 853,23
Update = 27100, Epoch = 50, LR = 0,000364, AvgCost = 199,7457, Sent = 592, SentPerMin = 832,45, WordPerSec = 763,62
Update = 27200, Epoch = 50, LR = 0,000364, AvgCost = 200,3932, Sent = 882, SentPerMin = 771,17, WordPerSec = 712,43
Update = 27238, Epoch = 50, LR = 0,000364, AvgCost = 198,9225, Sent = 1000, SentPerMin = 764,88, WordPerSec = 701,98
Update = 27300, Epoch = 51, LR = 0,000363, AvgCost = 201,4861, Sent = 178, SentPerMin = 932,88, WordPerSec = 860,82
Update = 27400, Epoch = 51, LR = 0,000362, AvgCost = 201,3726, Sent = 475, SentPerMin = 943,06, WordPerSec = 868,68
Update = 27500, Epoch = 51, LR = 0,000362, AvgCost = 200,2494, Sent = 771, SentPerMin = 944,16, WordPerSec = 872,68
Update = 27576, Epoch = 51, LR = 0,000361, AvgCost = 197,9492, Sent = 1000, SentPerMin = 956,46, WordPerSec = 877,80
Update = 27600, Epoch = 52, LR = 0,000361, AvgCost = 194,2494, Sent = 76, SentPerMin = 976,22, WordPerSec = 861,04
Update = 27700, Epoch = 52, LR = 0,000361, AvgCost = 201,3431, Sent = 363, SentPerMin = 939,44, WordPerSec = 866,29
Update = 27800, Epoch = 52, LR = 0,000360, AvgCost = 198,7797, Sent = 656, SentPerMin = 946,18, WordPerSec = 871,17
Update = 27900, Epoch = 52, LR = 0,000359, AvgCost = 196,8050, Sent = 961, SentPerMin = 956,46, WordPerSec = 876,65
Update = 27914, Epoch = 52, LR = 0,000359, AvgCost = 197,2202, Sent = 1000, SentPerMin = 956,41, WordPerSec = 877,76
Update = 28000, Epoch = 53, LR = 0,000359, AvgCost = 201,2065, Sent = 248, SentPerMin = 938,95, WordPerSec = 867,40
Update = 28100, Epoch = 53, LR = 0,000358, AvgCost = 197,8466, Sent = 549, SentPerMin = 950,33, WordPerSec = 870,73
Update = 28200, Epoch = 53, LR = 0,000357, AvgCost = 197,7684, Sent = 841, SentPerMin = 950,97, WordPerSec = 875,70
Update = 28252, Epoch = 53, LR = 0,000357, AvgCost = 196,6322, Sent = 1000, SentPerMin = 958,04, WordPerSec = 879,26
Update = 28300, Epoch = 54, LR = 0,000357, AvgCost = 195,7340, Sent = 140, SentPerMin = 945,68, WordPerSec = 848,86
Update = 28400, Epoch = 54, LR = 0,000356, AvgCost = 202,0472, Sent = 431, SentPerMin = 934,20, WordPerSec = 865,96
Update = 28500, Epoch = 54, LR = 0,000355, AvgCost = 195,8732, Sent = 732, SentPerMin = 951,85, WordPerSec = 870,23
Update = 28590, Epoch = 54, LR = 0,000355, AvgCost = 196,3412, Sent = 1000, SentPerMin = 954,06, WordPerSec = 875,60
Update = 28600, Epoch = 55, LR = 0,000355, AvgCost = 163,2234, Sent = 33, SentPerMin = 1046,59, WordPerSec = 808,73
Update = 28700, Epoch = 55, LR = 0,000354, AvgCost = 202,5791, Sent = 317, SentPerMin = 934,44, WordPerSec = 870,18
Update = 28800, Epoch = 55, LR = 0,000354, AvgCost = 197,0986, Sent = 618, SentPerMin = 948,21, WordPerSec = 870,24
Update = 28900, Epoch = 55, LR = 0,000353, AvgCost = 196,7905, Sent = 914, SentPerMin = 950,59, WordPerSec = 875,97
Update = 28928, Epoch = 55, LR = 0,000353, AvgCost = 195,8366, Sent = 1000, SentPerMin = 956,26, WordPerSec = 877,63
Update = 29000, Epoch = 56, LR = 0,000352, AvgCost = 203,5085, Sent = 205, SentPerMin = 925,62, WordPerSec = 864,21
Update = 29100, Epoch = 56, LR = 0,000352, AvgCost = 197,0944, Sent = 506, SentPerMin = 945,12, WordPerSec = 866,89
Update = 29200, Epoch = 56, LR = 0,000351, AvgCost = 196,9200, Sent = 802, SentPerMin = 943,42, WordPerSec = 868,54
Update = 29266, Epoch = 56, LR = 0,000351, AvgCost = 195,3276, Sent = 1000, SentPerMin = 944,09, WordPerSec = 866,46
Update = 29300, Epoch = 57, LR = 0,000351, AvgCost = 187,9912, Sent = 106, SentPerMin = 929,25, WordPerSec = 800,38
Update = 29400, Epoch = 57, LR = 0,000350, AvgCost = 198,2844, Sent = 393, SentPerMin = 903,87, WordPerSec = 833,03
Update = 29500, Epoch = 57, LR = 0,000349, AvgCost = 195,2670, Sent = 689, SentPerMin = 933,21, WordPerSec = 855,45
Update = 29600, Epoch = 57, LR = 0,000349, AvgCost = 195,1740, Sent = 989, SentPerMin = 941,99, WordPerSec = 864,76
Update = 29604, Epoch = 57, LR = 0,000349, AvgCost = 195,0393, Sent = 1000, SentPerMin = 942,11, WordPerSec = 864,64
Update = 29700, Epoch = 58, LR = 0,000348, AvgCost = 204,6459, Sent = 275, SentPerMin = 924,42, WordPerSec = 867,78
Update = 29800, Epoch = 58, LR = 0,000348, AvgCost = 195,8109, Sent = 581, SentPerMin = 945,08, WordPerSec = 866,97
Update = 29900, Epoch = 58, LR = 0,000347, AvgCost = 196,0503, Sent = 870, SentPerMin = 944,09, WordPerSec = 871,65
Update = 29942, Epoch = 58, LR = 0,000347, AvgCost = 194,7191, Sent = 1000, SentPerMin = 952,29, WordPerSec = 873,98
Update = 30000, Epoch = 59, LR = 0,000346, AvgCost = 201,4278, Sent = 166, SentPerMin = 665,67, WordPerSec = 613,40
Update = 30100, Epoch = 59, LR = 0,000346, AvgCost = 200,9535, Sent = 463, SentPerMin = 607,20, WordPerSec = 560,45
Update = 30200, Epoch = 59, LR = 0,000345, AvgCost = 197,9585, Sent = 761, SentPerMin = 707,39, WordPerSec = 651,40
Update = 30280, Epoch = 59, LR = 0,000345, AvgCost = 195,8277, Sent = 1000, SentPerMin = 758,31, WordPerSec = 695,95
Update = 30300, Epoch = 60, LR = 0,000345, AvgCost = 194,3662, Sent = 64, SentPerMin = 993,05, WordPerSec = 865,82
Update = 30400, Epoch = 60, LR = 0,000344, AvgCost = 199,8991, Sent = 352, SentPerMin = 939,97, WordPerSec = 864,40
Update = 30500, Epoch = 60, LR = 0,000344, AvgCost = 197,2711, Sent = 645, SentPerMin = 820,60, WordPerSec = 755,33
Update = 30600, Epoch = 60, LR = 0,000343, AvgCost = 196,2917, Sent = 948, SentPerMin = 772,22, WordPerSec = 709,90
Update = 30618, Epoch = 60, LR = 0,000343, AvgCost = 195,9019, Sent = 1000, SentPerMin = 767,67, WordPerSec = 704,54
Update = 30700, Epoch = 61, LR = 0,000342, AvgCost = 198,1016, Sent = 239, SentPerMin = 948,84, WordPerSec = 866,53
Update = 30800, Epoch = 61, LR = 0,000342, AvgCost = 197,7837, Sent = 536, SentPerMin = 945,08, WordPerSec = 868,70
Update = 30900, Epoch = 61, LR = 0,000341, AvgCost = 195,8152, Sent = 831, SentPerMin = 950,86, WordPerSec = 873,10
Update = 30956, Epoch = 61, LR = 0,000341, AvgCost = 195,1903, Sent = 1000, SentPerMin = 956,26, WordPerSec = 877,62
Update = 31000, Epoch = 62, LR = 0,000341, AvgCost = 194,0633, Sent = 131, SentPerMin = 957,99, WordPerSec = 852,56
Update = 31100, Epoch = 62, LR = 0,000340, AvgCost = 197,3411, Sent = 423, SentPerMin = 938,16, WordPerSec = 862,16
Update = 31200, Epoch = 62, LR = 0,000340, AvgCost = 194,3716, Sent = 720, SentPerMin = 951,05, WordPerSec = 870,37
Update = 31294, Epoch = 62, LR = 0,000339, AvgCost = 194,5594, Sent = 1000, SentPerMin = 953,32, WordPerSec = 874,93
Update = 31300, Epoch = 63, LR = 0,000339, AvgCost = 195,7881, Sent = 16, SentPerMin = 869,12, WordPerSec = 848,30
Update = 31400, Epoch = 63, LR = 0,000339, AvgCost = 202,4877, Sent = 304, SentPerMin = 927,87, WordPerSec = 866,62
Update = 31500, Epoch = 63, LR = 0,000338, AvgCost = 196,1484, Sent = 606, SentPerMin = 943,68, WordPerSec = 867,29
Update = 31600, Epoch = 63, LR = 0,000338, AvgCost = 195,3634, Sent = 902, SentPerMin = 948,49, WordPerSec = 873,55
Update = 31632, Epoch = 63, LR = 0,000337, AvgCost = 194,2335, Sent = 1000, SentPerMin = 954,41, WordPerSec = 875,93
Update = 31700, Epoch = 64, LR = 0,000337, AvgCost = 199,5356, Sent = 195, SentPerMin = 933,88, WordPerSec = 858,38
Update = 31800, Epoch = 64, LR = 0,000336, AvgCost = 196,0768, Sent = 495, SentPerMin = 947,80, WordPerSec = 866,20
Update = 31900, Epoch = 64, LR = 0,000336, AvgCost = 196,1155, Sent = 789, SentPerMin = 944,89, WordPerSec = 871,26
Update = 31970, Epoch = 64, LR = 0,000336, AvgCost = 193,8924, Sent = 1000, SentPerMin = 954,31, WordPerSec = 875,84
Update = 32000, Epoch = 65, LR = 0,000335, AvgCost = 192,6098, Sent = 92, SentPerMin = 957,91, WordPerSec = 850,15
Update = 32100, Epoch = 65, LR = 0,000335, AvgCost = 197,7850, Sent = 381, SentPerMin = 938,75, WordPerSec = 863,85
Update = 32200, Epoch = 65, LR = 0,000334, AvgCost = 195,0592, Sent = 676, SentPerMin = 946,01, WordPerSec = 868,51
Update = 32300, Epoch = 65, LR = 0,000334, AvgCost = 192,7659, Sent = 979, SentPerMin = 954,08, WordPerSec = 873,62
Update = 32308, Epoch = 65, LR = 0,000334, AvgCost = 193,7201, Sent = 1000, SentPerMin = 951,66, WordPerSec = 873,40
Starting inference...
Inference results:
••了了••
••••"•
••••"" (,     •••  ,, ,,    。 的"的。
••••"" ( (                                         的的的了。
••••••,

             */

            /*
             Update = 10100, Epoch = 0, LR = 0,000597, AvgCost = 265,9572, Sent = 285, SentPerMin = 603,01, WordPerSec = 566,23
Update = 10200, Epoch = 0, LR = 0,000594, AvgCost = 248,0596, Sent = 592, SentPerMin = 640,55, WordPerSec = 587,59
Update = 10300, Epoch = 0, LR = 0,000591, AvgCost = 245,8198, Sent = 882, SentPerMin = 647,79, WordPerSec = 598,44
Update = 10338, Epoch = 0, LR = 0,000590, AvgCost = 242,6293, Sent = 1000, SentPerMin = 654,36, WordPerSec = 600,55
Update = 10400, Epoch = 1, LR = 0,000588, AvgCost = 232,2862, Sent = 178, SentPerMin = 919,40, WordPerSec = 848,38
Update = 10500, Epoch = 1, LR = 0,000586, AvgCost = 233,5141, Sent = 475, SentPerMin = 928,84, WordPerSec = 855,58
Update = 10600, Epoch = 1, LR = 0,000583, AvgCost = 232,2139, Sent = 771, SentPerMin = 932,77, WordPerSec = 862,16
Update = 10676, Epoch = 1, LR = 0,000581, AvgCost = 229,0626, Sent = 1000, SentPerMin = 943,60, WordPerSec = 866,01
Update = 10700, Epoch = 2, LR = 0,000580, AvgCost = 221,5505, Sent = 76, SentPerMin = 935,30, WordPerSec = 824,95
Update = 10800, Epoch = 2, LR = 0,000577, AvgCost = 230,7410, Sent = 363, SentPerMin = 923,07, WordPerSec = 851,19
Update = 10900, Epoch = 2, LR = 0,000575, AvgCost = 227,5886, Sent = 656, SentPerMin = 934,19, WordPerSec = 860,14
Update = 11000, Epoch = 2, LR = 0,000572, AvgCost = 225,1032, Sent = 961, SentPerMin = 944,63, WordPerSec = 865,82
Update = 11014, Epoch = 2, LR = 0,000572, AvgCost = 225,5179, Sent = 1000, SentPerMin = 944,36, WordPerSec = 866,70
Update = 11100, Epoch = 3, LR = 0,000569, AvgCost = 228,2774, Sent = 248, SentPerMin = 918,22, WordPerSec = 848,24
Update = 11200, Epoch = 3, LR = 0,000567, AvgCost = 224,4254, Sent = 549, SentPerMin = 937,43, WordPerSec = 858,91
Update = 11300, Epoch = 3, LR = 0,000564, AvgCost = 224,5361, Sent = 841, SentPerMin = 937,67, WordPerSec = 863,45
Update = 11352, Epoch = 3, LR = 0,000563, AvgCost = 223,1930, Sent = 1000, SentPerMin = 943,95, WordPerSec = 866,32
Update = 11400, Epoch = 4, LR = 0,000562, AvgCost = 220,6293, Sent = 140, SentPerMin = 934,46, WordPerSec = 838,79
Update = 11500, Epoch = 4, LR = 0,000560, AvgCost = 228,0700, Sent = 431, SentPerMin = 917,30, WordPerSec = 850,29
Update = 11600, Epoch = 4, LR = 0,000557, AvgCost = 221,1198, Sent = 732, SentPerMin = 934,05, WordPerSec = 853,96
Update = 11690, Epoch = 4, LR = 0,000555, AvgCost = 221,5070, Sent = 1000, SentPerMin = 940,27, WordPerSec = 862,95
Update = 11700, Epoch = 5, LR = 0,000555, AvgCost = 185,0871, Sent = 33, SentPerMin = 1029,52, WordPerSec = 795,53
Update = 11800, Epoch = 5, LR = 0,000552, AvgCost = 227,8946, Sent = 317, SentPerMin = 927,14, WordPerSec = 863,38
Update = 11900, Epoch = 5, LR = 0,000550, AvgCost = 221,6512, Sent = 618, SentPerMin = 942,11, WordPerSec = 864,64
Update = 12000, Epoch = 5, LR = 0,000548, AvgCost = 221,4760, Sent = 914, SentPerMin = 945,34, WordPerSec = 871,13
Update = 12028, Epoch = 5, LR = 0,000547, AvgCost = 220,3402, Sent = 1000, SentPerMin = 950,49, WordPerSec = 872,33
Update = 12100, Epoch = 6, LR = 0,000545, AvgCost = 227,9361, Sent = 205, SentPerMin = 914,57, WordPerSec = 853,89
Update = 12200, Epoch = 6, LR = 0,000543, AvgCost = 221,0274, Sent = 506, SentPerMin = 935,06, WordPerSec = 857,66
Update = 12300, Epoch = 6, LR = 0,000541, AvgCost = 221,3601, Sent = 802, SentPerMin = 941,03, WordPerSec = 866,34
Update = 12366, Epoch = 6, LR = 0,000540, AvgCost = 219,6113, Sent = 1000, SentPerMin = 950,03, WordPerSec = 871,91
Update = 12400, Epoch = 7, LR = 0,000539, AvgCost = 209,5977, Sent = 106, SentPerMin = 978,60, WordPerSec = 842,89
Update = 12500, Epoch = 7, LR = 0,000537, AvgCost = 222,4796, Sent = 393, SentPerMin = 936,47, WordPerSec = 863,08
Update = 12600, Epoch = 7, LR = 0,000535, AvgCost = 219,6031, Sent = 689, SentPerMin = 950,69, WordPerSec = 871,46
Update = 12700, Epoch = 7, LR = 0,000532, AvgCost = 219,4339, Sent = 989, SentPerMin = 953,80, WordPerSec = 875,60
Update = 12704, Epoch = 7, LR = 0,000532, AvgCost = 219,2758, Sent = 1000, SentPerMin = 953,87, WordPerSec = 875,43
Update = 12800, Epoch = 8, LR = 0,000530, AvgCost = 229,2022, Sent = 275, SentPerMin = 920,39, WordPerSec = 864,00
Update = 12900, Epoch = 8, LR = 0,000528, AvgCost = 219,4717, Sent = 581, SentPerMin = 942,97, WordPerSec = 865,04
Update = 13000, Epoch = 8, LR = 0,000526, AvgCost = 220,1024, Sent = 870, SentPerMin = 910,41, WordPerSec = 840,56
Update = 13042, Epoch = 8, LR = 0,000525, AvgCost = 218,5922, Sent = 1000, SentPerMin = 919,65, WordPerSec = 844,03
Update = 13100, Epoch = 9, LR = 0,000524, AvgCost = 221,3100, Sent = 166, SentPerMin = 889,23, WordPerSec = 819,41
Update = 13200, Epoch = 9, LR = 0,000522, AvgCost = 221,6756, Sent = 463, SentPerMin = 904,53, WordPerSec = 834,89
Update = 13300, Epoch = 9, LR = 0,000520, AvgCost = 219,8132, Sent = 761, SentPerMin = 914,48, WordPerSec = 842,10
Update = 13380, Epoch = 9, LR = 0,000519, AvgCost = 217,7580, Sent = 1000, SentPerMin = 925,25, WordPerSec = 849,16
Update = 13400, Epoch = 10, LR = 0,000518, AvgCost = 214,3178, Sent = 64, SentPerMin = 720,81, WordPerSec = 628,45
Update = 13500, Epoch = 10, LR = 0,000516, AvgCost = 221,5309, Sent = 352, SentPerMin = 663,51, WordPerSec = 610,17
Update = 13600, Epoch = 10, LR = 0,000514, AvgCost = 219,3319, Sent = 645, SentPerMin = 662,32, WordPerSec = 609,64
Update = 13700, Epoch = 10, LR = 0,000513, AvgCost = 218,2119, Sent = 948, SentPerMin = 669,22, WordPerSec = 615,22
Update = 13718, Epoch = 10, LR = 0,000512, AvgCost = 217,7143, Sent = 1000, SentPerMin = 670,27, WordPerSec = 615,16
Update = 13800, Epoch = 11, LR = 0,000511, AvgCost = 219,4080, Sent = 239, SentPerMin = 947,51, WordPerSec = 865,32
Update = 13900, Epoch = 11, LR = 0,000509, AvgCost = 219,4018, Sent = 536, SentPerMin = 942,66, WordPerSec = 866,48
Update = 14000, Epoch = 11, LR = 0,000507, AvgCost = 217,9450, Sent = 831, SentPerMin = 943,99, WordPerSec = 866,80
Update = 14056, Epoch = 11, LR = 0,000506, AvgCost = 217,2746, Sent = 1000, SentPerMin = 948,66, WordPerSec = 870,65
Update = 14100, Epoch = 12, LR = 0,000505, AvgCost = 215,2656, Sent = 131, SentPerMin = 956,26, WordPerSec = 851,02
Update = 14200, Epoch = 12, LR = 0,000504, AvgCost = 218,6812, Sent = 423, SentPerMin = 939,81, WordPerSec = 863,68
Update = 14300, Epoch = 12, LR = 0,000502, AvgCost = 216,4061, Sent = 720, SentPerMin = 951,57, WordPerSec = 870,84
Update = 14394, Epoch = 12, LR = 0,000500, AvgCost = 216,6208, Sent = 1000, SentPerMin = 951,81, WordPerSec = 873,54
Update = 14400, Epoch = 13, LR = 0,000500, AvgCost = 217,0855, Sent = 16, SentPerMin = 865,81, WordPerSec = 845,06
Update = 14500, Epoch = 13, LR = 0,000498, AvgCost = 224,0097, Sent = 304, SentPerMin = 922,86, WordPerSec = 861,94
Update = 14600, Epoch = 13, LR = 0,000497, AvgCost = 217,5498, Sent = 606, SentPerMin = 939,90, WordPerSec = 863,82
Update = 14700, Epoch = 13, LR = 0,000495, AvgCost = 216,9777, Sent = 902, SentPerMin = 943,97, WordPerSec = 869,38
Update = 14732, Epoch = 13, LR = 0,000494, AvgCost = 215,8427, Sent = 1000, SentPerMin = 949,42, WordPerSec = 871,34
Update = 14800, Epoch = 14, LR = 0,000493, AvgCost = 220,9976, Sent = 195, SentPerMin = 935,51, WordPerSec = 859,87
Update = 14900, Epoch = 14, LR = 0,000492, AvgCost = 216,6673, Sent = 495, SentPerMin = 943,74, WordPerSec = 862,49
Update = 15000, Epoch = 14, LR = 0,000490, AvgCost = 217,4617, Sent = 789, SentPerMin = 939,18, WordPerSec = 865,99
Update = 15070, Epoch = 14, LR = 0,000489, AvgCost = 215,0931, Sent = 1000, SentPerMin = 948,56, WordPerSec = 870,55
Update = 15100, Epoch = 15, LR = 0,000488, AvgCost = 212,2576, Sent = 92, SentPerMin = 955,56, WordPerSec = 848,06
Update = 15200, Epoch = 15, LR = 0,000487, AvgCost = 218,0848, Sent = 381, SentPerMin = 936,65, WordPerSec = 861,92
Update = 15300, Epoch = 15, LR = 0,000485, AvgCost = 215,6586, Sent = 676, SentPerMin = 945,63, WordPerSec = 868,16
Update = 15400, Epoch = 15, LR = 0,000483, AvgCost = 213,3145, Sent = 979, SentPerMin = 953,93, WordPerSec = 873,48
Update = 15408, Epoch = 15, LR = 0,000483, AvgCost = 214,3923, Sent = 1000, SentPerMin = 951,68, WordPerSec = 873,42
Update = 15500, Epoch = 16, LR = 0,000482, AvgCost = 221,1057, Sent = 265, SentPerMin = 928,14, WordPerSec = 861,36
Update = 15600, Epoch = 16, LR = 0,000480, AvgCost = 215,7710, Sent = 566, SentPerMin = 940,62, WordPerSec = 865,73
Update = 15700, Epoch = 16, LR = 0,000479, AvgCost = 214,9856, Sent = 859, SentPerMin = 942,13, WordPerSec = 868,17
Update = 15746, Epoch = 16, LR = 0,000478, AvgCost = 213,7380, Sent = 1000, SentPerMin = 938,80, WordPerSec = 861,60
Update = 15800, Epoch = 17, LR = 0,000477, AvgCost = 215,9926, Sent = 157, SentPerMin = 893,24, WordPerSec = 812,55
Update = 15900, Epoch = 17, LR = 0,000476, AvgCost = 217,7018, Sent = 450, SentPerMin = 919,38, WordPerSec = 847,78
Update = 16000, Epoch = 17, LR = 0,000474, AvgCost = 213,2593, Sent = 752, SentPerMin = 932,36, WordPerSec = 854,81
Update = 16084, Epoch = 17, LR = 0,000473, AvgCost = 213,2631, Sent = 1000, SentPerMin = 936,96, WordPerSec = 859,91
Update = 16100, Epoch = 18, LR = 0,000473, AvgCost = 206,4916, Sent = 51, SentPerMin = 946,44, WordPerSec = 815,92
Update = 16200, Epoch = 18, LR = 0,000471, AvgCost = 218,8293, Sent = 336, SentPerMin = 896,48, WordPerSec = 830,58
Update = 16300, Epoch = 18, LR = 0,000470, AvgCost = 213,2518, Sent = 636, SentPerMin = 886,10, WordPerSec = 811,86
Update = 16400, Epoch = 18, LR = 0,000469, AvgCost = 213,2973, Sent = 935, SentPerMin = 869,95, WordPerSec = 799,70
Update = 16422, Epoch = 18, LR = 0,000468, AvgCost = 212,8994, Sent = 1000, SentPerMin = 870,21, WordPerSec = 798,65
Update = 16500, Epoch = 19, LR = 0,000467, AvgCost = 217,1191, Sent = 225, SentPerMin = 534,58, WordPerSec = 491,42
Update = 16600, Epoch = 19, LR = 0,000466, AvgCost = 215,5311, Sent = 524, SentPerMin = 557,63, WordPerSec = 513,09
Update = 16700, Epoch = 19, LR = 0,000464, AvgCost = 214,1239, Sent = 819, SentPerMin = 571,22, WordPerSec = 525,41
Update = 16760, Epoch = 19, LR = 0,000463, AvgCost = 212,9474, Sent = 1000, SentPerMin = 576,32, WordPerSec = 528,92
Update = 16800, Epoch = 20, LR = 0,000463, AvgCost = 208,4972, Sent = 122, SentPerMin = 866,84, WordPerSec = 756,59
Update = 16900, Epoch = 20, LR = 0,000462, AvgCost = 215,3720, Sent = 411, SentPerMin = 833,26, WordPerSec = 767,44
Update = 17000, Epoch = 20, LR = 0,000460, AvgCost = 211,7855, Sent = 709, SentPerMin = 855,27, WordPerSec = 781,24
Update = 17098, Epoch = 20, LR = 0,000459, AvgCost = 212,1804, Sent = 1000, SentPerMin = 847,94, WordPerSec = 778,21
Update = 17100, Epoch = 21, LR = 0,000459, AvgCost = 188,5047, Sent = 6, SentPerMin = 813,68, WordPerSec = 739,09
Update = 17200, Epoch = 21, LR = 0,000457, AvgCost = 220,4058, Sent = 292, SentPerMin = 858,18, WordPerSec = 802,93
Update = 17300, Epoch = 21, LR = 0,000456, AvgCost = 213,3459, Sent = 595, SentPerMin = 868,90, WordPerSec = 799,73
Update = 17400, Epoch = 21, LR = 0,000455, AvgCost = 212,7504, Sent = 890, SentPerMin = 871,24, WordPerSec = 803,13
Update = 17436, Epoch = 21, LR = 0,000454, AvgCost = 211,3267, Sent = 1000, SentPerMin = 872,11, WordPerSec = 800,40
Update = 17500, Epoch = 22, LR = 0,000454, AvgCost = 217,8259, Sent = 184, SentPerMin = 796,97, WordPerSec = 732,21
Update = 17600, Epoch = 22, LR = 0,000452, AvgCost = 213,6350, Sent = 482, SentPerMin = 820,36, WordPerSec = 754,21
Update = 17700, Epoch = 22, LR = 0,000451, AvgCost = 213,9951, Sent = 775, SentPerMin = 826,68, WordPerSec = 765,94
Update = 17774, Epoch = 22, LR = 0,000450, AvgCost = 210,7334, Sent = 1000, SentPerMin = 850,25, WordPerSec = 780,33
Update = 17800, Epoch = 23, LR = 0,000450, AvgCost = 205,0359, Sent = 81, SentPerMin = 921,47, WordPerSec = 817,95
Update = 17900, Epoch = 23, LR = 0,000448, AvgCost = 215,2239, Sent = 368, SentPerMin = 905,08, WordPerSec = 837,61
Update = 18000, Epoch = 23, LR = 0,000447, AvgCost = 211,1728, Sent = 663, SentPerMin = 895,69, WordPerSec = 822,94
Update = 18100, Epoch = 23, LR = 0,000446, AvgCost = 208,9506, Sent = 969, SentPerMin = 886,49, WordPerSec = 810,64
Update = 18112, Epoch = 23, LR = 0,000446, AvgCost = 209,8988, Sent = 1000, SentPerMin = 884,30, WordPerSec = 811,58
Update = 18200, Epoch = 24, LR = 0,000445, AvgCost = 219,7130, Sent = 251, SentPerMin = 853,38, WordPerSec = 797,22
Update = 18300, Epoch = 24, LR = 0,000444, AvgCost = 210,3461, Sent = 556, SentPerMin = 874,35, WordPerSec = 800,30
Update = 18400, Epoch = 24, LR = 0,000442, AvgCost = 210,8631, Sent = 847, SentPerMin = 873,50, WordPerSec = 805,11
Update = 18450, Epoch = 24, LR = 0,000442, AvgCost = 209,5250, Sent = 1000, SentPerMin = 880,37, WordPerSec = 807,97
Update = 18500, Epoch = 25, LR = 0,000441, AvgCost = 212,1511, Sent = 145, SentPerMin = 875,16, WordPerSec = 795,99
Update = 18600, Epoch = 25, LR = 0,000440, AvgCost = 214,9052, Sent = 437, SentPerMin = 868,25, WordPerSec = 804,60
Update = 18700, Epoch = 25, LR = 0,000439, AvgCost = 208,9938, Sent = 739, SentPerMin = 883,73, WordPerSec = 810,10
Update = 18788, Epoch = 25, LR = 0,000438, AvgCost = 209,0111, Sent = 1000, SentPerMin = 891,24, WordPerSec = 817,95
Update = 18800, Epoch = 26, LR = 0,000438, AvgCost = 169,9526, Sent = 41, SentPerMin = 1022,02, WordPerSec = 786,88
Update = 18900, Epoch = 26, LR = 0,000436, AvgCost = 215,3988, Sent = 324, SentPerMin = 878,19, WordPerSec = 815,72
Update = 19000, Epoch = 26, LR = 0,000435, AvgCost = 209,8501, Sent = 624, SentPerMin = 885,13, WordPerSec = 811,44
Update = 19100, Epoch = 26, LR = 0,000434, AvgCost = 209,4496, Sent = 921, SentPerMin = 886,53, WordPerSec = 815,65
Update = 19126, Epoch = 26, LR = 0,000434, AvgCost = 208,6225, Sent = 1000, SentPerMin = 889,80, WordPerSec = 816,63
Update = 19200, Epoch = 27, LR = 0,000433, AvgCost = 213,8605, Sent = 213, SentPerMin = 882,55, WordPerSec = 814,18
Update = 19300, Epoch = 27, LR = 0,000432, AvgCost = 211,6259, Sent = 511, SentPerMin = 881,14, WordPerSec = 812,51
Update = 19400, Epoch = 27, LR = 0,000431, AvgCost = 209,8421, Sent = 808, SentPerMin = 890,41, WordPerSec = 820,18
Update = 19464, Epoch = 27, LR = 0,000430, AvgCost = 208,0337, Sent = 1000, SentPerMin = 894,33, WordPerSec = 820,78
Update = 19500, Epoch = 28, LR = 0,000430, AvgCost = 197,8339, Sent = 113, SentPerMin = 665,25, WordPerSec = 569,88
Update = 19600, Epoch = 28, LR = 0,000429, AvgCost = 211,5484, Sent = 399, SentPerMin = 619,56, WordPerSec = 570,28
Update = 19700, Epoch = 28, LR = 0,000427, AvgCost = 208,9164, Sent = 696, SentPerMin = 610,84, WordPerSec = 559,66
Update = 19800, Epoch = 28, LR = 0,000426, AvgCost = 208,6872, Sent = 995, SentPerMin = 623,41, WordPerSec = 571,75
Update = 19802, Epoch = 28, LR = 0,000426, AvgCost = 208,6338, Sent = 1000, SentPerMin = 623,16, WordPerSec = 571,91
Update = 19900, Epoch = 29, LR = 0,000425, AvgCost = 217,4048, Sent = 280, SentPerMin = 894,76, WordPerSec = 839,59
Update = 20000, Epoch = 29, LR = 0,000424, AvgCost = 207,9256, Sent = 588, SentPerMin = 890,18, WordPerSec = 814,29
Update = 20100, Epoch = 29, LR = 0,000423, AvgCost = 208,8285, Sent = 877, SentPerMin = 688,87, WordPerSec = 635,32
Update = 20140, Epoch = 29, LR = 0,000423, AvgCost = 207,5948, Sent = 1000, SentPerMin = 711,41, WordPerSec = 652,91
Update = 20200, Epoch = 30, LR = 0,000422, AvgCost = 209,6188, Sent = 173, SentPerMin = 885,79, WordPerSec = 814,96
Update = 20300, Epoch = 30, LR = 0,000421, AvgCost = 210,6364, Sent = 469, SentPerMin = 891,71, WordPerSec = 822,54
Update = 20400, Epoch = 30, LR = 0,000420, AvgCost = 209,1806, Sent = 766, SentPerMin = 900,29, WordPerSec = 830,73
Update = 20478, Epoch = 30, LR = 0,000419, AvgCost = 206,9503, Sent = 1000, SentPerMin = 913,59, WordPerSec = 838,46
Update = 20500, Epoch = 31, LR = 0,000419, AvgCost = 198,6102, Sent = 71, SentPerMin = 957,68, WordPerSec = 828,86
Update = 20600, Epoch = 31, LR = 0,000418, AvgCost = 209,7821, Sent = 358, SentPerMin = 904,57, WordPerSec = 830,91
Update = 20700, Epoch = 31, LR = 0,000417, AvgCost = 207,3063, Sent = 652, SentPerMin = 908,59, WordPerSec = 834,34
Update = 20800, Epoch = 31, LR = 0,000416, AvgCost = 206,2972, Sent = 955, SentPerMin = 909,44, WordPerSec = 834,35
Update = 20816, Epoch = 31, LR = 0,000416, AvgCost = 206,2658, Sent = 1000, SentPerMin = 909,09, WordPerSec = 834,34
Update = 20900, Epoch = 32, LR = 0,000415, AvgCost = 207,2595, Sent = 244, SentPerMin = 898,28, WordPerSec = 820,66
Update = 21000, Epoch = 32, LR = 0,000414, AvgCost = 207,2130, Sent = 542, SentPerMin = 886,12, WordPerSec = 813,86
Update = 21100, Epoch = 32, LR = 0,000413, AvgCost = 206,7923, Sent = 835, SentPerMin = 889,89, WordPerSec = 818,71
Update = 21154, Epoch = 32, LR = 0,000413, AvgCost = 205,5861, Sent = 1000, SentPerMin = 896,40, WordPerSec = 822,69
Update = 21200, Epoch = 33, LR = 0,000412, AvgCost = 202,6741, Sent = 136, SentPerMin = 898,72, WordPerSec = 799,70
Update = 21300, Epoch = 33, LR = 0,000411, AvgCost = 207,6789, Sent = 428, SentPerMin = 882,68, WordPerSec = 811,87
Update = 21400, Epoch = 33, LR = 0,000410, AvgCost = 205,2890, Sent = 725, SentPerMin = 892,15, WordPerSec = 817,80
Update = 21492, Epoch = 33, LR = 0,000409, AvgCost = 205,1169, Sent = 1000, SentPerMin = 891,75, WordPerSec = 818,42
Update = 21500, Epoch = 34, LR = 0,000409, AvgCost = 185,9761, Sent = 23, SentPerMin = 868,30, WordPerSec = 756,93
Update = 21600, Epoch = 34, LR = 0,000408, AvgCost = 210,8244, Sent = 311, SentPerMin = 874,27, WordPerSec = 812,85
Update = 21700, Epoch = 34, LR = 0,000407, AvgCost = 206,7581, Sent = 611, SentPerMin = 889,60, WordPerSec = 819,35
Update = 21800, Epoch = 34, LR = 0,000406, AvgCost = 206,1688, Sent = 906, SentPerMin = 896,59, WordPerSec = 827,10
Update = 21830, Epoch = 34, LR = 0,000406, AvgCost = 204,7446, Sent = 1000, SentPerMin = 904,07, WordPerSec = 829,73
Update = 21900, Epoch = 35, LR = 0,000405, AvgCost = 208,4545, Sent = 200, SentPerMin = 871,90, WordPerSec = 800,12
Update = 22000, Epoch = 35, LR = 0,000405, AvgCost = 205,7106, Sent = 501, SentPerMin = 902,83, WordPerSec = 827,11
Update = 22100, Epoch = 35, LR = 0,000404, AvgCost = 206,4481, Sent = 795, SentPerMin = 903,89, WordPerSec = 832,84
Update = 22168, Epoch = 35, LR = 0,000403, AvgCost = 204,4444, Sent = 1000, SentPerMin = 908,36, WordPerSec = 833,66
Update = 22200, Epoch = 36, LR = 0,000403, AvgCost = 198,0261, Sent = 99, SentPerMin = 923,49, WordPerSec = 804,71
Update = 22300, Epoch = 36, LR = 0,000402, AvgCost = 205,7426, Sent = 388, SentPerMin = 901,18, WordPerSec = 825,96
Update = 22400, Epoch = 36, LR = 0,000401, AvgCost = 204,5346, Sent = 683, SentPerMin = 909,45, WordPerSec = 833,33
Update = 22500, Epoch = 36, LR = 0,000400, AvgCost = 204,1940, Sent = 984, SentPerMin = 917,00, WordPerSec = 841,53
Update = 22506, Epoch = 36, LR = 0,000400, AvgCost = 204,0546, Sent = 1000, SentPerMin = 916,64, WordPerSec = 841,26
Update = 22600, Epoch = 37, LR = 0,000399, AvgCost = 212,9792, Sent = 269, SentPerMin = 613,33, WordPerSec = 573,28
Update = 22700, Epoch = 37, LR = 0,000398, AvgCost = 206,5186, Sent = 573, SentPerMin = 615,46, WordPerSec = 565,25
Update = 22800, Epoch = 37, LR = 0,000397, AvgCost = 206,3577, Sent = 865, SentPerMin = 619,33, WordPerSec = 570,43
Update = 22844, Epoch = 37, LR = 0,000397, AvgCost = 205,2857, Sent = 1000, SentPerMin = 622,24, WordPerSec = 571,07
Update = 22900, Epoch = 38, LR = 0,000396, AvgCost = 207,8144, Sent = 161, SentPerMin = 829,75, WordPerSec = 761,12
Update = 23000, Epoch = 38, LR = 0,000396, AvgCost = 208,5323, Sent = 456, SentPerMin = 843,97, WordPerSec = 779,81
Update = 23100, Epoch = 38, LR = 0,000395, AvgCost = 205,3589, Sent = 757, SentPerMin = 846,96, WordPerSec = 779,05
Update = 23182, Epoch = 38, LR = 0,000394, AvgCost = 204,2056, Sent = 1000, SentPerMin = 851,79, WordPerSec = 781,75
Update = 23200, Epoch = 39, LR = 0,000394, AvgCost = 191,3133, Sent = 59, SentPerMin = 938,10, WordPerSec = 783,87
Update = 23300, Epoch = 39, LR = 0,000393, AvgCost = 208,2050, Sent = 343, SentPerMin = 838,55, WordPerSec = 775,31
Update = 23400, Epoch = 39, LR = 0,000392, AvgCost = 204,7876, Sent = 640, SentPerMin = 844,57, WordPerSec = 776,10
Update = 23500, Epoch = 39, LR = 0,000391, AvgCost = 203,6213, Sent = 943, SentPerMin = 854,77, WordPerSec = 784,58
Update = 23520, Epoch = 39, LR = 0,000391, AvgCost = 203,4763, Sent = 1000, SentPerMin = 852,64, WordPerSec = 782,52
Update = 23600, Epoch = 40, LR = 0,000391, AvgCost = 206,0828, Sent = 231, SentPerMin = 730,54, WordPerSec = 672,24
Update = 23700, Epoch = 40, LR = 0,000390, AvgCost = 205,5889, Sent = 529, SentPerMin = 741,40, WordPerSec = 682,37
Update = 23800, Epoch = 40, LR = 0,000389, AvgCost = 204,1811, Sent = 824, SentPerMin = 757,29, WordPerSec = 696,85
Update = 23858, Epoch = 40, LR = 0,000388, AvgCost = 202,7542, Sent = 1000, SentPerMin = 773,37, WordPerSec = 709,77
Update = 23900, Epoch = 41, LR = 0,000388, AvgCost = 200,6216, Sent = 126, SentPerMin = 785,56, WordPerSec = 693,49
Update = 24000, Epoch = 41, LR = 0,000387, AvgCost = 204,6225, Sent = 418, SentPerMin = 760,70, WordPerSec = 698,04
Update = 24100, Epoch = 41, LR = 0,000386, AvgCost = 201,8711, Sent = 715, SentPerMin = 790,95, WordPerSec = 723,32
Update = 24196, Epoch = 41, LR = 0,000386, AvgCost = 202,0664, Sent = 1000, SentPerMin = 789,69, WordPerSec = 724,75
Update = 24200, Epoch = 42, LR = 0,000386, AvgCost = 181,6994, Sent = 12, SentPerMin = 672,19, WordPerSec = 604,97
Update = 24300, Epoch = 42, LR = 0,000385, AvgCost = 209,4679, Sent = 299, SentPerMin = 749,64, WordPerSec = 699,25
Update = 24400, Epoch = 42, LR = 0,000384, AvgCost = 203,3484, Sent = 601, SentPerMin = 794,74, WordPerSec = 730,08
Update = 24500, Epoch = 42, LR = 0,000383, AvgCost = 202,4736, Sent = 897, SentPerMin = 788,44, WordPerSec = 725,23
Update = 24534, Epoch = 42, LR = 0,000383, AvgCost = 201,4305, Sent = 1000, SentPerMin = 790,23, WordPerSec = 725,24
Update = 24600, Epoch = 43, LR = 0,000383, AvgCost = 207,4283, Sent = 189, SentPerMin = 755,92, WordPerSec = 696,32
Update = 24700, Epoch = 43, LR = 0,000382, AvgCost = 202,9081, Sent = 490, SentPerMin = 764,70, WordPerSec = 699,08
Update = 24800, Epoch = 43, LR = 0,000381, AvgCost = 203,4460, Sent = 784, SentPerMin = 782,71, WordPerSec = 721,36
Update = 24872, Epoch = 43, LR = 0,000380, AvgCost = 201,0037, Sent = 1000, SentPerMin = 789,53, WordPerSec = 724,60
Update = 24900, Epoch = 44, LR = 0,000380, AvgCost = 195,5014, Sent = 87, SentPerMin = 835,11, WordPerSec = 733,84
Update = 25000, Epoch = 44, LR = 0,000379, AvgCost = 204,4435, Sent = 375, SentPerMin = 785,09, WordPerSec = 723,61
Update = 25100, Epoch = 44, LR = 0,000379, AvgCost = 202,4035, Sent = 668, SentPerMin = 789,10, WordPerSec = 727,15
Update = 25200, Epoch = 44, LR = 0,000378, AvgCost = 199,4951, Sent = 974, SentPerMin = 800,53, WordPerSec = 732,55
Update = 25210, Epoch = 44, LR = 0,000378, AvgCost = 200,4457, Sent = 1000, SentPerMin = 797,80, WordPerSec = 732,19
Update = 25300, Epoch = 45, LR = 0,000377, AvgCost = 210,5592, Sent = 258, SentPerMin = 594,08, WordPerSec = 553,75
Update = 25400, Epoch = 45, LR = 0,000376, AvgCost = 202,9601, Sent = 562, SentPerMin = 630,00, WordPerSec = 577,67
Update = 25500, Epoch = 45, LR = 0,000376, AvgCost = 203,1142, Sent = 854, SentPerMin = 628,33, WordPerSec = 578,62
Update = 25548, Epoch = 45, LR = 0,000375, AvgCost = 201,7774, Sent = 1000, SentPerMin = 632,00, WordPerSec = 580,03
Update = 25600, Epoch = 46, LR = 0,000375, AvgCost = 201,1033, Sent = 152, SentPerMin = 889,21, WordPerSec = 799,99
Update = 25700, Epoch = 46, LR = 0,000374, AvgCost = 205,9854, Sent = 443, SentPerMin = 879,58, WordPerSec = 813,23
Update = 25800, Epoch = 46, LR = 0,000374, AvgCost = 201,3824, Sent = 744, SentPerMin = 895,69, WordPerSec = 822,64
Update = 25886, Epoch = 46, LR = 0,000373, AvgCost = 200,9331, Sent = 1000, SentPerMin = 901,17, WordPerSec = 827,06
Update = 25900, Epoch = 47, LR = 0,000373, AvgCost = 189,7199, Sent = 45, SentPerMin = 930,85, WordPerSec = 762,27
Update = 26000, Epoch = 47, LR = 0,000372, AvgCost = 206,6379, Sent = 330, SentPerMin = 884,91, WordPerSec = 820,28
Update = 26100, Epoch = 47, LR = 0,000371, AvgCost = 201,6415, Sent = 630, SentPerMin = 894,99, WordPerSec = 820,76
Update = 26200, Epoch = 47, LR = 0,000371, AvgCost = 200,7926, Sent = 929, SentPerMin = 897,86, WordPerSec = 824,91
Update = 26224, Epoch = 47, LR = 0,000371, AvgCost = 200,2395, Sent = 1000, SentPerMin = 900,51, WordPerSec = 826,46
Update = 26300, Epoch = 48, LR = 0,000370, AvgCost = 204,8056, Sent = 219, SentPerMin = 887,53, WordPerSec = 817,82
Update = 26400, Epoch = 48, LR = 0,000369, AvgCost = 202,5102, Sent = 519, SentPerMin = 885,02, WordPerSec = 812,12
Update = 26500, Epoch = 48, LR = 0,000369, AvgCost = 201,2182, Sent = 814, SentPerMin = 888,36, WordPerSec = 817,12
Update = 26562, Epoch = 48, LR = 0,000368, AvgCost = 199,7570, Sent = 1000, SentPerMin = 892,04, WordPerSec = 818,68
Update = 26600, Epoch = 49, LR = 0,000368, AvgCost = 186,3955, Sent = 119, SentPerMin = 944,25, WordPerSec = 803,28
Update = 26700, Epoch = 49, LR = 0,000367, AvgCost = 202,1432, Sent = 406, SentPerMin = 890,79, WordPerSec = 818,72
Update = 26800, Epoch = 49, LR = 0,000367, AvgCost = 199,2988, Sent = 702, SentPerMin = 899,67, WordPerSec = 822,77
Update = 26900, Epoch = 49, LR = 0,000366, AvgCost = 199,1516, Sent = 1000, SentPerMin = 897,42, WordPerSec = 823,62
Update = 26900, Epoch = 49, LR = 0,000366, AvgCost = 199,1516, Sent = 1000, SentPerMin = 897,41, WordPerSec = 823,61
Update = 27000, Epoch = 50, LR = 0,000365, AvgCost = 209,1686, Sent = 285, SentPerMin = 896,96, WordPerSec = 842,25
Update = 27100, Epoch = 50, LR = 0,000364, AvgCost = 199,9380, Sent = 592, SentPerMin = 920,73, WordPerSec = 844,60
Update = 27200, Epoch = 50, LR = 0,000364, AvgCost = 200,5383, Sent = 882, SentPerMin = 914,61, WordPerSec = 844,94
Update = 27238, Epoch = 50, LR = 0,000364, AvgCost = 198,7247, Sent = 1000, SentPerMin = 921,77, WordPerSec = 845,97
Update = 27300, Epoch = 51, LR = 0,000363, AvgCost = 201,7435, Sent = 178, SentPerMin = 885,32, WordPerSec = 816,94
Update = 27400, Epoch = 51, LR = 0,000362, AvgCost = 202,2312, Sent = 475, SentPerMin = 897,60, WordPerSec = 826,80
Update = 27500, Epoch = 51, LR = 0,000362, AvgCost = 201,0933, Sent = 771, SentPerMin = 903,48, WordPerSec = 835,08
Update = 27576, Epoch = 51, LR = 0,000361, AvgCost = 198,4987, Sent = 1000, SentPerMin = 913,43, WordPerSec = 838,32
Update = 27600, Epoch = 52, LR = 0,000361, AvgCost = 194,9233, Sent = 76, SentPerMin = 931,26, WordPerSec = 821,38
Update = 27700, Epoch = 52, LR = 0,000361, AvgCost = 203,1501, Sent = 363, SentPerMin = 885,01, WordPerSec = 816,09
Update = 27800, Epoch = 52, LR = 0,000360, AvgCost = 200,0578, Sent = 656, SentPerMin = 891,67, WordPerSec = 820,99
Update = 27900, Epoch = 52, LR = 0,000359, AvgCost = 197,9511, Sent = 961, SentPerMin = 901,30, WordPerSec = 826,10
Update = 27914, Epoch = 52, LR = 0,000359, AvgCost = 198,2657, Sent = 1000, SentPerMin = 901,81, WordPerSec = 827,65
Update = 28000, Epoch = 53, LR = 0,000359, AvgCost = 202,8300, Sent = 248, SentPerMin = 895,46, WordPerSec = 827,22
Update = 28100, Epoch = 53, LR = 0,000358, AvgCost = 199,0467, Sent = 549, SentPerMin = 899,27, WordPerSec = 823,95
Update = 28200, Epoch = 53, LR = 0,000357, AvgCost = 199,0033, Sent = 841, SentPerMin = 900,08, WordPerSec = 828,84
Update = 28252, Epoch = 53, LR = 0,000357, AvgCost = 197,7232, Sent = 1000, SentPerMin = 908,11, WordPerSec = 833,43
Update = 28300, Epoch = 54, LR = 0,000357, AvgCost = 198,4059, Sent = 140, SentPerMin = 657,06, WordPerSec = 589,79
Update = 28400, Epoch = 54, LR = 0,000356, AvgCost = 205,1736, Sent = 431, SentPerMin = 639,77, WordPerSec = 593,04
Update = 28500, Epoch = 54, LR = 0,000355, AvgCost = 198,4737, Sent = 732, SentPerMin = 649,67, WordPerSec = 593,96
Update = 28590, Epoch = 54, LR = 0,000355, AvgCost = 198,9763, Sent = 1000, SentPerMin = 625,12, WordPerSec = 573,71
Update = 28600, Epoch = 55, LR = 0,000355, AvgCost = 164,4930, Sent = 33, SentPerMin = 800,52, WordPerSec = 618,59
Update = 28700, Epoch = 55, LR = 0,000354, AvgCost = 205,5353, Sent = 317, SentPerMin = 706,08, WordPerSec = 657,52
Update = 28800, Epoch = 55, LR = 0,000354, AvgCost = 199,5443, Sent = 618, SentPerMin = 732,31, WordPerSec = 672,09
Update = 28900, Epoch = 55, LR = 0,000353, AvgCost = 199,1081, Sent = 914, SentPerMin = 733,98, WordPerSec = 676,37
Update = 28928, Epoch = 55, LR = 0,000353, AvgCost = 198,0247, Sent = 1000, SentPerMin = 741,01, WordPerSec = 680,08
Update = 29000, Epoch = 56, LR = 0,000352, AvgCost = 205,7955, Sent = 205, SentPerMin = 757,66, WordPerSec = 707,39
Update = 29100, Epoch = 56, LR = 0,000352, AvgCost = 199,2024, Sent = 506, SentPerMin = 752,97, WordPerSec = 690,64
Update = 29200, Epoch = 56, LR = 0,000351, AvgCost = 199,0138, Sent = 802, SentPerMin = 749,34, WordPerSec = 689,87
Update = 29266, Epoch = 56, LR = 0,000351, AvgCost = 197,2619, Sent = 1000, SentPerMin = 770,03, WordPerSec = 706,71
Update = 29300, Epoch = 57, LR = 0,000351, AvgCost = 188,9001, Sent = 106, SentPerMin = 979,02, WordPerSec = 843,25
Update = 29400, Epoch = 57, LR = 0,000350, AvgCost = 200,4918, Sent = 393, SentPerMin = 930,15, WordPerSec = 857,26
Update = 29500, Epoch = 57, LR = 0,000349, AvgCost = 197,1457, Sent = 689, SentPerMin = 946,89, WordPerSec = 867,98
Update = 29600, Epoch = 57, LR = 0,000349, AvgCost = 196,8508, Sent = 989, SentPerMin = 947,88, WordPerSec = 870,17
Update = 29604, Epoch = 57, LR = 0,000349, AvgCost = 196,7017, Sent = 1000, SentPerMin = 947,91, WordPerSec = 869,96
Update = 29700, Epoch = 58, LR = 0,000348, AvgCost = 207,0161, Sent = 275, SentPerMin = 918,24, WordPerSec = 861,98
Update = 29800, Epoch = 58, LR = 0,000348, AvgCost = 197,5391, Sent = 581, SentPerMin = 940,66, WordPerSec = 862,92
Update = 29900, Epoch = 58, LR = 0,000347, AvgCost = 197,7306, Sent = 870, SentPerMin = 939,88, WordPerSec = 867,77
Update = 29942, Epoch = 58, LR = 0,000347, AvgCost = 196,3185, Sent = 1000, SentPerMin = 948,03, WordPerSec = 870,07
Update = 30000, Epoch = 59, LR = 0,000346, AvgCost = 200,6665, Sent = 166, SentPerMin = 919,34, WordPerSec = 847,16
Update = 30100, Epoch = 59, LR = 0,000346, AvgCost = 200,0328, Sent = 463, SentPerMin = 591,67, WordPerSec = 546,12
Update = 30200, Epoch = 59, LR = 0,000345, AvgCost = 197,6352, Sent = 761, SentPerMin = 693,43, WordPerSec = 638,55
Update = 30280, Epoch = 59, LR = 0,000345, AvgCost = 195,8373, Sent = 1000, SentPerMin = 745,28, WordPerSec = 683,99
Update = 30300, Epoch = 60, LR = 0,000345, AvgCost = 192,3856, Sent = 64, SentPerMin = 980,75, WordPerSec = 855,09
Update = 30400, Epoch = 60, LR = 0,000344, AvgCost = 199,8799, Sent = 352, SentPerMin = 932,12, WordPerSec = 857,18
Update = 30500, Epoch = 60, LR = 0,000344, AvgCost = 196,9152, Sent = 645, SentPerMin = 934,87, WordPerSec = 860,52
Update = 30600, Epoch = 60, LR = 0,000343, AvgCost = 196,0681, Sent = 948, SentPerMin = 943,35, WordPerSec = 867,22
Update = 30618, Epoch = 60, LR = 0,000343, AvgCost = 195,6111, Sent = 1000, SentPerMin = 945,76, WordPerSec = 867,98
Update = 30700, Epoch = 61, LR = 0,000342, AvgCost = 197,6946, Sent = 239, SentPerMin = 931,50, WordPerSec = 850,69
Update = 30800, Epoch = 61, LR = 0,000342, AvgCost = 197,3765, Sent = 536, SentPerMin = 931,82, WordPerSec = 856,52
Update = 30900, Epoch = 61, LR = 0,000341, AvgCost = 195,7682, Sent = 831, SentPerMin = 938,37, WordPerSec = 861,64
Update = 30956, Epoch = 61, LR = 0,000341, AvgCost = 195,1789, Sent = 1000, SentPerMin = 943,91, WordPerSec = 866,29
Update = 31000, Epoch = 62, LR = 0,000341, AvgCost = 193,8897, Sent = 131, SentPerMin = 956,89, WordPerSec = 851,58
Update = 31100, Epoch = 62, LR = 0,000340, AvgCost = 197,3005, Sent = 423, SentPerMin = 934,77, WordPerSec = 859,05
Update = 31200, Epoch = 62, LR = 0,000340, AvgCost = 194,4245, Sent = 720, SentPerMin = 945,31, WordPerSec = 865,11
Update = 31294, Epoch = 62, LR = 0,000339, AvgCost = 194,7297, Sent = 1000, SentPerMin = 946,76, WordPerSec = 868,90
Update = 31300, Epoch = 63, LR = 0,000339, AvgCost = 193,5209, Sent = 16, SentPerMin = 604,48, WordPerSec = 589,99
Update = 31400, Epoch = 63, LR = 0,000339, AvgCost = 203,7373, Sent = 304, SentPerMin = 656,88, WordPerSec = 613,52
Update = 31500, Epoch = 63, LR = 0,000338, AvgCost = 197,3738, Sent = 606, SentPerMin = 664,25, WordPerSec = 610,49
Update = 31600, Epoch = 63, LR = 0,000338, AvgCost = 196,9263, Sent = 902, SentPerMin = 665,14, WordPerSec = 612,58
Update = 31632, Epoch = 63, LR = 0,000337, AvgCost = 195,8159, Sent = 1000, SentPerMin = 668,93, WordPerSec = 613,92
Update = 31700, Epoch = 64, LR = 0,000337, AvgCost = 200,4316, Sent = 195, SentPerMin = 924,90, WordPerSec = 850,12
Update = 31800, Epoch = 64, LR = 0,000336, AvgCost = 196,3583, Sent = 495, SentPerMin = 941,29, WordPerSec = 860,25
Update = 31900, Epoch = 64, LR = 0,000336, AvgCost = 196,7958, Sent = 789, SentPerMin = 935,06, WordPerSec = 862,19
Update = 31970, Epoch = 64, LR = 0,000336, AvgCost = 194,7533, Sent = 1000, SentPerMin = 944,25, WordPerSec = 866,60
Update = 32000, Epoch = 65, LR = 0,000335, AvgCost = 193,0374, Sent = 92, SentPerMin = 954,88, WordPerSec = 847,46
Update = 32100, Epoch = 65, LR = 0,000335, AvgCost = 198,0336, Sent = 381, SentPerMin = 932,79, WordPerSec = 858,37
Update = 32200, Epoch = 65, LR = 0,000334, AvgCost = 195,3565, Sent = 676, SentPerMin = 942,53, WordPerSec = 865,31
Update = 32300, Epoch = 65, LR = 0,000334, AvgCost = 193,4480, Sent = 979, SentPerMin = 951,49, WordPerSec = 871,24
Update = 32308, Epoch = 65, LR = 0,000334, AvgCost = 194,4262, Sent = 1000, SentPerMin = 949,35, WordPerSec = 871,28
Update = 32400, Epoch = 66, LR = 0,000333, AvgCost = 201,1427, Sent = 265, SentPerMin = 924,99, WordPerSec = 858,44
Update = 32500, Epoch = 66, LR = 0,000333, AvgCost = 196,0322, Sent = 566, SentPerMin = 936,66, WordPerSec = 862,08
Update = 32600, Epoch = 66, LR = 0,000332, AvgCost = 195,1750, Sent = 859, SentPerMin = 938,56, WordPerSec = 864,88
Update = 32646, Epoch = 66, LR = 0,000332, AvgCost = 193,9869, Sent = 1000, SentPerMin = 946,32, WordPerSec = 868,50
Update = 32700, Epoch = 67, LR = 0,000332, AvgCost = 196,7458, Sent = 157, SentPerMin = 933,05, WordPerSec = 848,76
Update = 32800, Epoch = 67, LR = 0,000331, AvgCost = 197,9486, Sent = 450, SentPerMin = 934,96, WordPerSec = 862,13
Update = 32900, Epoch = 67, LR = 0,000331, AvgCost = 193,4770, Sent = 752, SentPerMin = 941,71, WordPerSec = 863,38
Update = 32984, Epoch = 67, LR = 0,000330, AvgCost = 193,5788, Sent = 1000, SentPerMin = 947,40, WordPerSec = 869,49
Update = 33000, Epoch = 68, LR = 0,000330, AvgCost = 187,7161, Sent = 51, SentPerMin = 982,67, WordPerSec = 847,15
Update = 33100, Epoch = 68, LR = 0,000330, AvgCost = 199,3403, Sent = 336, SentPerMin = 928,61, WordPerSec = 860,35
Update = 33200, Epoch = 68, LR = 0,000329, AvgCost = 193,8361, Sent = 636, SentPerMin = 941,93, WordPerSec = 863,02
Update = 33300, Epoch = 68, LR = 0,000329, AvgCost = 193,7468, Sent = 935, SentPerMin = 945,69, WordPerSec = 869,32
Update = 33322, Epoch = 68, LR = 0,000329, AvgCost = 193,3266, Sent = 1000, SentPerMin = 949,45, WordPerSec = 871,37
Update = 33400, Epoch = 69, LR = 0,000328, AvgCost = 196,6394, Sent = 225, SentPerMin = 934,70, WordPerSec = 859,23
Update = 33500, Epoch = 69, LR = 0,000328, AvgCost = 195,5881, Sent = 524, SentPerMin = 933,66, WordPerSec = 859,09
Update = 33600, Epoch = 69, LR = 0,000327, AvgCost = 193,8813, Sent = 819, SentPerMin = 936,53, WordPerSec = 861,42
Update = 33660, Epoch = 69, LR = 0,000327, AvgCost = 192,9154, Sent = 1000, SentPerMin = 943,52, WordPerSec = 865,94
Update = 33700, Epoch = 70, LR = 0,000327, AvgCost = 189,0945, Sent = 122, SentPerMin = 969,83, WordPerSec = 846,48
Update = 33800, Epoch = 70, LR = 0,000326, AvgCost = 195,3912, Sent = 411, SentPerMin = 931,02, WordPerSec = 857,47
Update = 33900, Epoch = 70, LR = 0,000326, AvgCost = 191,8918, Sent = 709, SentPerMin = 947,44, WordPerSec = 865,44
Update = 33998, Epoch = 70, LR = 0,000325, AvgCost = 192,5274, Sent = 1000, SentPerMin = 946,32, WordPerSec = 868,50
Update = 34000, Epoch = 71, LR = 0,000325, AvgCost = 167,6259, Sent = 6, SentPerMin = 924,12, WordPerSec = 839,41
Update = 34100, Epoch = 71, LR = 0,000325, AvgCost = 200,6219, Sent = 292, SentPerMin = 917,01, WordPerSec = 857,97
Update = 34200, Epoch = 71, LR = 0,000324, AvgCost = 194,2094, Sent = 595, SentPerMin = 933,82, WordPerSec = 859,48
Update = 34300, Epoch = 71, LR = 0,000324, AvgCost = 193,6143, Sent = 890, SentPerMin = 939,38, WordPerSec = 865,93
Update = 34336, Epoch = 71, LR = 0,000324, AvgCost = 192,3915, Sent = 1000, SentPerMin = 943,35, WordPerSec = 865,77
Update = 34400, Epoch = 72, LR = 0,000323, AvgCost = 198,4969, Sent = 184, SentPerMin = 914,50, WordPerSec = 840,19
Update = 34500, Epoch = 72, LR = 0,000323, AvgCost = 194,7687, Sent = 482, SentPerMin = 929,91, WordPerSec = 854,93
Update = 34600, Epoch = 72, LR = 0,000323, AvgCost = 194,8227, Sent = 775, SentPerMin = 927,34, WordPerSec = 859,19
Update = 34674, Epoch = 72, LR = 0,000322, AvgCost = 192,1438, Sent = 1000, SentPerMin = 940,71, WordPerSec = 863,35
Update = 34700, Epoch = 73, LR = 0,000322, AvgCost = 188,3442, Sent = 81, SentPerMin = 680,97, WordPerSec = 604,46
Update = 34800, Epoch = 73, LR = 0,000322, AvgCost = 198,3047, Sent = 368, SentPerMin = 654,19, WordPerSec = 605,42
Update = 34900, Epoch = 73, LR = 0,000321, AvgCost = 194,5431, Sent = 663, SentPerMin = 661,37, WordPerSec = 607,65
Update = 35000, Epoch = 73, LR = 0,000321, AvgCost = 192,5562, Sent = 969, SentPerMin = 669,46, WordPerSec = 612,18
Update = 35012, Epoch = 73, LR = 0,000321, AvgCost = 193,4560, Sent = 1000, SentPerMin = 667,38, WordPerSec = 612,50
Update = 35100, Epoch = 74, LR = 0,000320, AvgCost = 201,6635, Sent = 251, SentPerMin = 818,02, WordPerSec = 764,20
Update = 35200, Epoch = 74, LR = 0,000320, AvgCost = 193,3788, Sent = 556, SentPerMin = 777,02, WordPerSec = 711,22
Update = 35300, Epoch = 74, LR = 0,000319, AvgCost = 193,8119, Sent = 847, SentPerMin = 756,38, WordPerSec = 697,16
Update = 35350, Epoch = 74, LR = 0,000319, AvgCost = 192,6551, Sent = 1000, SentPerMin = 757,56, WordPerSec = 695,26
Update = 35400, Epoch = 75, LR = 0,000319, AvgCost = 194,8117, Sent = 145, SentPerMin = 715,23, WordPerSec = 650,53
Update = 35500, Epoch = 75, LR = 0,000318, AvgCost = 197,1636, Sent = 437, SentPerMin = 715,73, WordPerSec = 663,26
Update = 35600, Epoch = 75, LR = 0,000318, AvgCost = 191,7857, Sent = 739, SentPerMin = 725,13, WordPerSec = 664,72
Update = 35688, Epoch = 75, LR = 0,000318, AvgCost = 192,0020, Sent = 1000, SentPerMin = 728,68, WordPerSec = 668,76
Update = 35700, Epoch = 76, LR = 0,000318, AvgCost = 156,1881, Sent = 41, SentPerMin = 813,86, WordPerSec = 626,61
Update = 35800, Epoch = 76, LR = 0,000317, AvgCost = 197,8026, Sent = 324, SentPerMin = 710,99, WordPerSec = 660,41
Update = 35900, Epoch = 76, LR = 0,000317, AvgCost = 192,5146, Sent = 624, SentPerMin = 723,19, WordPerSec = 662,98
Update = 36000, Epoch = 76, LR = 0,000316, AvgCost = 192,3200, Sent = 921, SentPerMin = 726,99, WordPerSec = 668,87
Update = 36026, Epoch = 76, LR = 0,000316, AvgCost = 191,6906, Sent = 1000, SentPerMin = 729,68, WordPerSec = 669,67
Update = 36100, Epoch = 77, LR = 0,000316, AvgCost = 196,6969, Sent = 213, SentPerMin = 709,62, WordPerSec = 654,65
Update = 36200, Epoch = 77, LR = 0,000315, AvgCost = 194,5298, Sent = 511, SentPerMin = 715,75, WordPerSec = 660,00
Update = 36300, Epoch = 77, LR = 0,000315, AvgCost = 192,9526, Sent = 808, SentPerMin = 722,74, WordPerSec = 665,73
Update = 36364, Epoch = 77, LR = 0,000315, AvgCost = 191,4537, Sent = 1000, SentPerMin = 728,06, WordPerSec = 668,19
Update = 36400, Epoch = 78, LR = 0,000314, AvgCost = 181,1235, Sent = 113, SentPerMin = 761,43, WordPerSec = 652,27
Update = 36500, Epoch = 78, LR = 0,000314, AvgCost = 193,8347, Sent = 399, SentPerMin = 721,12, WordPerSec = 663,77
Update = 36600, Epoch = 78, LR = 0,000314, AvgCost = 191,4283, Sent = 696, SentPerMin = 729,06, WordPerSec = 667,98
Update = 36700, Epoch = 78, LR = 0,000313, AvgCost = 191,1104, Sent = 995, SentPerMin = 731,86, WordPerSec = 671,21
Update = 36702, Epoch = 78, LR = 0,000313, AvgCost = 191,0616, Sent = 1000, SentPerMin = 731,36, WordPerSec = 671,22
Update = 36800, Epoch = 79, LR = 0,000313, AvgCost = 200,1218, Sent = 280, SentPerMin = 703,52, WordPerSec = 660,14
Update = 36900, Epoch = 79, LR = 0,000312, AvgCost = 191,3287, Sent = 588, SentPerMin = 758,52, WordPerSec = 693,85
Update = 37000, Epoch = 79, LR = 0,000312, AvgCost = 191,7809, Sent = 877, SentPerMin = 786,48, WordPerSec = 725,34
Update = 37040, Epoch = 79, LR = 0,000312, AvgCost = 190,7298, Sent = 1000, SentPerMin = 800,23, WordPerSec = 734,43
Starting inference...
Inference results:
,的的在在的
了了在在  的
,   了了 (          和和和和和和和和了了"。
和 的了了的了的了了。。
了了和了 的了了在

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
                MaxEpochNum = 80, // 66, // 53, // 40, // 26, // 20, // 13, // 3,
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
