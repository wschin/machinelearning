// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using Google.Protobuf;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Command;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model.Pmf;
using Newtonsoft.Json;

[assembly: LoadableClass(SavePmfCommand.Summary, typeof(SavePmfCommand), typeof(SavePmfCommand.Arguments), typeof(SignatureCommand),
    "Save PMF", "SavePmf", DocName = "command/SavePmf.md")]

[assembly: LoadableClass(typeof(void), typeof(SavePmfCommand), null, typeof(SignatureEntryPointModule), "SavePmf")]


namespace Microsoft.ML.Runtime.Model.Pmf
{
    public sealed class SavePmfCommand : DataCommand.ImplBase<SavePmfCommand.Arguments>
    {
        public const string Summary = "Given a data model, write out the corresponding PMF (protable model format).";
        public const string LoadName = "SavePmf";

        public sealed class Arguments : DataCommand.ArgumentsBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "The path to write the output PMF to.", SortOrder = 1)]
            public string PmfPath;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The path to write the output JSON to.", SortOrder = 2)]
            public string JsonPath;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The 'name' property in the output ONNX. By default this will be the PMF extension-less name.", NullName = "<Auto>", SortOrder = 3)]
            public string Name;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The 'domain' property in the output ONNX.", NullName = "<Auto>", SortOrder = 4)]
            public string Domain;

            [Argument(ArgumentType.AtMostOnce, Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly, HelpText = "Comma delimited list of input column names to drop", ShortName = "idrop", SortOrder = 5)]
            public string InputsToDrop;

            [Argument(ArgumentType.AtMostOnce, Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly, HelpText = "Comma delimited list of output column names to drop", ShortName = "odrop", SortOrder = 7)]
            public string OutputsToDrop;

            [Argument(ArgumentType.AtMostOnce, Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly, HelpText = "Whether we should attempt to load the predictor and attach the scorer to the pipeline if one is present.", ShortName = "pred", SortOrder = 9)]
            public bool? LoadPredictor;

            [Argument(ArgumentType.Required, Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly, HelpText = "Model that needs to be converted to Pmf format.", SortOrder = 10)]
            public ITransformModel Model;
        }

        private readonly string _outputModelPath;
        private readonly string _outputJsonModelPath;
        private readonly string _name;
        private readonly string _domain;
        private readonly bool? _loadPredictor;
        private readonly HashSet<string> _inputsToDrop;
        private readonly HashSet<string> _outputsToDrop;
        private readonly ITransformModel _model;
        private const string ProducerName = "ML.NET";
        private const long ModelVersion = 0;

        public SavePmfCommand(IHostEnvironment env, Arguments args)
                : base(env, args, LoadName)
        {
            Host.CheckValue(args, nameof(args));
            Utils.CheckOptionalUserDirectory(args.PmfPath, nameof(args.JsonPath));
            _outputModelPath = string.IsNullOrWhiteSpace(args.PmfPath) ? null : args.PmfPath;
            _outputJsonModelPath = string.IsNullOrWhiteSpace(args.JsonPath) ? null : args.JsonPath;
            if (args.Name == null && _outputModelPath != null)
                _name = Path.GetFileNameWithoutExtension(_outputModelPath);
            else if (!string.IsNullOrWhiteSpace(args.Name))
                _name = args.Name;

            _loadPredictor = args.LoadPredictor;
            _inputsToDrop = CreateDropMap(args.InputsToDrop?.Split(','));
            _outputsToDrop = CreateDropMap(args.OutputsToDrop?.Split(','));
            _domain = args.Domain;
            _model = args.Model;
        }

        private static HashSet<string> CreateDropMap(string[] toDrop)
        {
            if (toDrop == null)
                return new HashSet<string>();

            return new HashSet<string>(toDrop);
        }

        public override void Run()
        {
            using (var ch = Host.Start("Run"))
            {
                Run(ch);
                ch.Done();
            }
        }


        private void Run(IChannel ch)
        {
            System.Console.Write("This is my command");
        }

        public sealed class Output
        {
        }

        [TlcModule.EntryPoint(Name = "Models.PmfConverter", Desc = "Converts the model to PMF format.", UserName = "PMF Converter.")]
        public static Output Apply(IHostEnvironment env, Arguments input)
        {
            new SavePmfCommand(env, input).Run();
            return new Output();
        }
    }
}
