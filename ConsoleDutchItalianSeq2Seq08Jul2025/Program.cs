using Seq2SeqSharp;
using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Enums;
using Seq2SeqSharp.LearningRate;
using Seq2SeqSharp.Metrics;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;

// ChatGPT, modified
namespace ConsoleDutchItalianSeq2Seq08Jul2025
{
    internal class Program
    {
        static void Main(string[] args)
        {
            string srcLang = "NL";
            string tgtLang = "IT";
            string modelFilePath = "nl2it.model";

            var trainData = new List<(string src, string tgt)>
            {
                ("Hallo , hoe gaat het ?", "Ciao , come stai ?"),
                ("Ik hou van jou", "Ti amo"),
                ("Dit is een boek", "Questo è un libro"),
                ("Zij is een lerares", "Lei è un'insegnante"),
                ("Wat is jouw naam ?", "Come ti chiami ?"),
                ("Dit is mijn huis", "Questa è la mia casa"),
                ("Ik spreek een beetje Engels", "Parlo un po' di inglese"),
                ("Waar is het station ?", "Dove è la stazione ?"),
                ("Hoe laat is het ?", "Che ore sono ?"),
                ("Ik ben moe", "Sono stanco"),
                ("Wat doe je ?", "Cosa fai ?"),
                ("Ik kom uit Nederland", "Vengo dai Paesi Bassi"),
                ("Ik heb honger", "Ho fame"),
                ("Ik heb dorst", "Ho sete"),
                ("Waar woon je ?", "Dove vivi ?"),
                ("Ik werk als ingenieur", "Lavoro come ingegnere"),
                ("Zij is mijn zus", "Lei è mia sorella"),
                ("Hij is mijn broer", "Lui è mio fratello"),
                ("Ik wil koffie", "Voglio del caffè"),
                ("Wil je thee ?", "Vuoi del tè ?"),
            };

            string srcTrainFile = "train.nl.snt"; // Do not change file extension.
            string tgtTrainFile = "train.it.snt";

            File.WriteAllLines(srcTrainFile, trainData.ConvertAll(p => p.src));
            File.WriteAllLines(tgtTrainFile, trainData.ConvertAll(p => p.tgt));

            string rootPath = Directory.GetCurrentDirectory();

            var opts = new Seq2SeqOptions
            {
                Task = ModeEnums.Train,
                SrcLang = srcLang,
                TgtLang = tgtLang,
                EncoderLayerDepth = 2,
                DecoderLayerDepth = 2,
                HiddenSize = 512,
                SrcEmbeddingDim = 512,
                TgtEmbeddingDim = 512,
                MaxEpochNum = 200,
                MaxTokenSizePerBatch = 10,
                ValMaxTokenSizePerBatch = 10,
                MaxSrcSentLength = 20,
                MaxTgtSentLength = 20,
                PaddingType = PaddingEnums.AllowPadding,
                TooLongSequence = TooLongSequence.Ignore,
                ProcessorType = ProcessorTypeEnums.CPU,
                ModelFilePath = modelFilePath,
                StartLearningRate = 0.001f,
                WarmUpSteps = 10,
                SharedEmbeddings = false,
                EncoderType = EncoderTypeEnums.BiLSTM,
                DecoderType = DecoderTypeEnums.AttentionLSTM,
                TrainCorpusPath = rootPath
            };

            var trainCorpus = new Seq2SeqCorpus(
                corpusFilePath: opts.TrainCorpusPath,
                srcLangName: srcLang,
                tgtLangName: tgtLang,
                maxTokenSizePerBatch: opts.MaxTokenSizePerBatch,
                maxSrcSentLength: opts.MaxSrcSentLength,
                maxTgtSentLength: opts.MaxTgtSentLength,
                paddingEnums: opts.PaddingType,
                tooLongSequence: opts.TooLongSequence);

            var (srcVocab, tgtVocab) = trainCorpus.BuildVocabs(1000, 1000, false);
            var learningRate = new DecayLearningRate(opts.StartLearningRate, opts.WarmUpSteps, 0, 0.7f, 100);
            var optimizer = Misc.CreateOptimizer(opts);
            var metrics = new List<IMetric> { new BleuMetric() };

            var model = new Seq2Seq(opts, srcVocab, tgtVocab);
            model.StatusUpdateWatcher += (s, e) =>
            {
                if (e is CostEventArg cost)
                {
                    Console.WriteLine($"Epoch {cost.Epoch}, Update {cost.Update}, Cost = {cost.AvgCostInTotal:F4}");
                }
            };

            model.Train(
                maxTrainingEpoch: opts.MaxEpochNum,
                trainCorpus: trainCorpus,
                validCorpusList: Array.Empty<Seq2SeqCorpus>(),
                learningRate: learningRate,
                optimizer: optimizer,
                metrics: metrics.ToArray(),
                decodingOptions: opts.CreateDecodingOptions());

            model.SaveModel(suffix: ".trained");

            // Inference
            opts.Task = ModeEnums.Test;
            opts.ModelFilePath = modelFilePath + ".trained";
            var inferModel = new Seq2Seq(opts);

            string testInputPath = "test_input.nl.snt";
            string testOutputPath = "test_output.it.snt";
            File.WriteAllLines(testInputPath, new[]
            {
                "Zij is mijn zus",
                "Wat is jouw naam ?",
                "Wat doe je ?",
                "Hallo , hoe gaat het ?",
                "Wil je koffie ?",
                "Ik wil thee"
            });

            inferModel.Test(
                inputTestFile: testInputPath,
                outputFile: testOutputPath,
                batchSize: 1,
                decodingOptions: opts.CreateDecodingOptions(),
                srcSpmPath: null,
                tgtSpmPath: null); // We are not using SentencePiece

            Console.WriteLine("\nTranslations:");
            foreach (var line in File.ReadLines(testOutputPath))
            {
                Console.WriteLine(line);
            }

            string[] files1 = Directory.GetFiles(rootPath, "*.tmp.sorted.txt");

            foreach (string file in files1)
            {
                File.Delete(file);
            }

            string[] files2 = Directory.GetFiles(rootPath, "nl2it.model.*");
            foreach (string file in files2)
            {
                File.Delete(file);
            }

            File.Delete(srcTrainFile);
            File.Delete(tgtTrainFile);
            File.Delete(testInputPath);
            File.Delete(testOutputPath);

            /*
 Epoch 9, Update 100, Cost = 3,3633
Epoch 18, Update 200, Cost = 0,9186
Epoch 27, Update 300, Cost = 0,1396
Epoch 36, Update 400, Cost = 0,0740
Epoch 45, Update 500, Cost = 0,0404
Epoch 54, Update 600, Cost = 0,0208
Epoch 63, Update 700, Cost = 0,0120
Epoch 72, Update 800, Cost = 0,0073
Epoch 81, Update 900, Cost = 0,0050
Epoch 90, Update 1000, Cost = 3,0251
Epoch 99, Update 1100, Cost = 1,4446
Epoch 109, Update 1200, Cost = 2,0901
Epoch 118, Update 1300, Cost = 2,3536
Epoch 127, Update 1400, Cost = 1,5377
Epoch 136, Update 1500, Cost = 1,1647
Epoch 145, Update 1600, Cost = 0,8835
Epoch 154, Update 1700, Cost = 0,7508
Epoch 163, Update 1800, Cost = 0,6422
Epoch 172, Update 1900, Cost = 0,5112
Epoch 181, Update 2000, Cost = 0,4468
Epoch 190, Update 2100, Cost = 0,4000
Epoch 199, Update 2200, Cost = 0,3610

Translations:
<s> Lei è mia sorella </s>
<s> Come ti chiami ? </s>
<s> Cosa fai ? </s>
<s> Ciao , come stai ? </s>
<s> Vuoi del tè ? </s>
<s> Voglio del caffè </s>
             */

            /*
             Epoch 9, Update 100, Cost = 3,6078
Epoch 18, Update 200, Cost = 0,6032
Epoch 27, Update 300, Cost = 0,1412
Epoch 36, Update 400, Cost = 0,0744
Epoch 45, Update 500, Cost = 0,0403
Epoch 54, Update 600, Cost = 0,0206
Epoch 63, Update 700, Cost = 0,0121
Epoch 72, Update 800, Cost = 0,0075
Epoch 81, Update 900, Cost = 0,0051
Epoch 90, Update 1000, Cost = 4,1063
Epoch 99, Update 1100, Cost = 1,7812
Epoch 109, Update 1200, Cost = 3,0779
Epoch 118, Update 1300, Cost = 2,0590
Epoch 127, Update 1400, Cost = 1,5601
Epoch 136, Update 1500, Cost = 1,2936
Epoch 145, Update 1600, Cost = 0,9827
Epoch 154, Update 1700, Cost = 0,8215
Epoch 163, Update 1800, Cost = 0,7067
Epoch 172, Update 1900, Cost = 0,5403
Epoch 181, Update 2000, Cost = 0,4721
Epoch 190, Update 2100, Cost = 0,4225
Epoch 199, Update 2200, Cost = 0,3821

Translations:
<s> Lei è mia sorella </s>
<s> Come ti chiami ? </s>
<s> Cosa fai ? </s>
<s> Ciao , come stai ? </s>
<s> Vuoi del tè ? </s>
<s> Voglio del caffè </s>
             */


            /*
             Epoch 9, Update 100, Cost = 3,8330
Epoch 18, Update 200, Cost = 0,5936
Epoch 27, Update 300, Cost = 0,1458
Epoch 36, Update 400, Cost = 0,0762
Epoch 45, Update 500, Cost = 0,0443
Epoch 54, Update 600, Cost = 0,0235
Epoch 63, Update 700, Cost = 0,0137
Epoch 72, Update 800, Cost = 0,0086
Epoch 81, Update 900, Cost = 0,0059
Epoch 90, Update 1000, Cost = 4,3996
Epoch 99, Update 1100, Cost = 1,9315
Epoch 109, Update 1200, Cost = 3,7138
Epoch 118, Update 1300, Cost = 2,5535
Epoch 127, Update 1400, Cost = 1,5849
Epoch 136, Update 1500, Cost = 1,3438
Epoch 145, Update 1600, Cost = 1,0625
Epoch 154, Update 1700, Cost = 0,9269
Epoch 163, Update 1800, Cost = 0,7789
Epoch 172, Update 1900, Cost = 0,5941
Epoch 181, Update 2000, Cost = 0,5217
Epoch 190, Update 2100, Cost = 0,4723
Epoch 199, Update 2200, Cost = 0,4458

Translations:
<s> Lei è mia sorella </s>
<s> Come ti chiami ? </s>
<s> Cosa fai ? </s>
<s> Ciao , come stai ? </s>
<s> Cosa fai ? ? </s>
<s> Ho del caffè </s>
             */


            Console.ReadLine();
        }
    }
}
