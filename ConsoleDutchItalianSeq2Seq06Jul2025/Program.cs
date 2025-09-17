using Seq2SeqSharp;
using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Enums;
using Seq2SeqSharp.LearningRate;
using Seq2SeqSharp.Metrics;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;

// ChatGPT, modified
namespace ConsoleDutchItalianSeq2Seq06Jul2025
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
                ("Ik ben moe", "Sono stanco")
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
                HiddenSize = 256,
                SrcEmbeddingDim = 256,
                TgtEmbeddingDim = 256,
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
                "Hoe laat is het ?",
                "Dit is mijn huis",
                "Ik hou van mijn lerares en mijn boek"
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
   Epoch 16, Update 100, Cost = 0,5523
Epoch 33, Update 200, Cost = 0,0970
Epoch 49, Update 300, Cost = 0,0461
Epoch 66, Update 400, Cost = 0,0263
Epoch 83, Update 500, Cost = 0,0177
Epoch 99, Update 600, Cost = 0,0107
Epoch 116, Update 700, Cost = 0,0075
Epoch 133, Update 800, Cost = 0,0061
Epoch 149, Update 900, Cost = 0,0045
Epoch 166, Update 1000, Cost = 0,0039
Epoch 183, Update 1100, Cost = 0,0038
Epoch 199, Update 1200, Cost = 0,0032

Translations:
<s> Che ore sono ? </s>
<s> Questa è la mia casa </s>
<s> Questa stanco </s>
             */

            /*
             Epoch 16, Update 100, Cost = 0,2822
Epoch 33, Update 200, Cost = 0,0834
Epoch 49, Update 300, Cost = 0,0407
Epoch 66, Update 400, Cost = 0,0220
Epoch 83, Update 500, Cost = 0,0149
Epoch 99, Update 600, Cost = 0,0094
Epoch 116, Update 700, Cost = 0,0062
Epoch 133, Update 800, Cost = 0,0051
Epoch 149, Update 900, Cost = 0,0040
Epoch 166, Update 1000, Cost = 0,0032
Epoch 183, Update 1100, Cost = 0,0032
Epoch 199, Update 1200, Cost = 0,0028

Translations:
<s> Che ore sono ? </s>
<s> Questa è la mia casa </s>
<s> Ti stanco </s>
             */

            /*
             Epoch 16, Update 100, Cost = 0,2730
Epoch 33, Update 200, Cost = 0,0818
Epoch 49, Update 300, Cost = 0,0393
Epoch 66, Update 400, Cost = 0,0222
Epoch 83, Update 500, Cost = 0,0147
Epoch 99, Update 600, Cost = 0,0091
Epoch 116, Update 700, Cost = 0,0064
Epoch 133, Update 800, Cost = 0,0052
Epoch 149, Update 900, Cost = 0,0039
Epoch 166, Update 1000, Cost = 0,0034
Epoch 183, Update 1100, Cost = 0,0033
Epoch 199, Update 1200, Cost = 0,0028

Translations:
<s> Che ore sono ? </s>
<s> Questa è la mia casa </s>
<s> Questa è </s>
             */

            Console.ReadLine();
        }
    }
}
