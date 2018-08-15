// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using LotusvNext;
using LotusvNext.Expressions;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Runtime.Model.Pmf
{
    /// <summary>
    /// A context for defining a ONNX output.
    /// </summary>
    public class PmfContext
    {
        private Dictionary<string, string> _ourColumnNameToPmfNameMap;
        private Dictionary<string, string> _ourOperatorNameToPmfNameMap;
        // All existing variable names. New variables must not exist in this set.
        private HashSet<string> _variableNamePool;
        // All existing node names. New node names must not alrady exist in this set.
        private HashSet<string> _operatorNamePool;
        private Dictionary<string, LotusvNext.Types.TypeProto> _modelInputParameters;
        private Dictionary<string, LotusvNext.Types.TypeProto> _modelOutputParameters;
        private Dictionary<string, LotusvNext.Types.TypeProto> _varTypes;
        private Dictionary<string, LotusvNext.Expressions.Expression> _refPool;
        private Dictionary<string, LotusvNext.Expressions.Expression> _defPool;
        Dictionary<string, LotusvNext.FunctionDefProto> _functions;
        private List<LotusvNext.Expressions.Expression> _expressions;

        public PmfContext()
        {
            _variableNamePool = new HashSet<string>();
            _operatorNamePool = new HashSet<string>();
            _ourColumnNameToPmfNameMap = new Dictionary<string, string>();
            _ourOperatorNameToPmfNameMap = new Dictionary<string, string>();
            _modelInputParameters = new Dictionary<string, LotusvNext.Types.TypeProto>();
            _modelOutputParameters = new Dictionary<string, LotusvNext.Types.TypeProto>();
            _varTypes = new Dictionary<string, LotusvNext.Types.TypeProto>();
            _functions = new Dictionary<string, LotusvNext.FunctionDefProto>();
            _expressions = new List<LotusvNext.Expressions.Expression>();
            _refPool = new Dictionary<string, Expression>();
            _defPool = new Dictionary<string, Expression>();
        }

        private string CreateUniqueString(ref HashSet<string> pool, string prefix)
        {
            if (!pool.Contains(prefix))
            {
                pool.Add(prefix);
                return prefix;
            }

            // The string pool contains the input prefix, so we are going to append something to make a unique string.
 
            // The index will be appended
            int count = 1;
            // Candidate string
            string derivedString = prefix + count;;
            // Keep increasing count until prefix + count cannot be found in the pool
            while(pool.Contains(derivedString))
            {
                derivedString = prefix + ++count;
            }
            // Add derived_string into the pool so that we will not declare the same string again in the future
            pool.Add(derivedString);
            return derivedString;
        }
        public string CreateOperatorName(string prefix)
        {
            string name = CreateUniqueString(ref _variableNamePool, prefix);
            _ourOperatorNameToPmfNameMap[prefix] = name;
            return name;
        }

        public string CreateVariableName(string prefix)
        {
            string name = CreateUniqueString(ref _operatorNamePool, prefix);
            _ourColumnNameToPmfNameMap[prefix] = name;
            _refPool[name] = PmfUtils.MakeVarRef(name);
            return name;
        }

        public string RetrieveVariableNameOrCreateOne(string prefix)
        {
            if (_ourColumnNameToPmfNameMap.ContainsKey(prefix))
                return _ourColumnNameToPmfNameMap[prefix];
            else
                return CreateVariableName(prefix);
        }

        // Use to create I/O for FunctionalModelProto
        public void AddInputVariable(ColumnType columnType, string colName)
        {
            // Declare name in IR
            var name = CreateVariableName(colName);
            // Create TypeProto according to the input column
            var typeProto = PmfUtils.MakeType(columnType);
            // Create ParameterDeclProto based on information above
            _modelInputParameters[name] = typeProto;
            _refPool[name] = PmfUtils.MakeVarRef(name);
        }

        public void AddOutputVariable(ColumnType columnType, string colName)
        {
            var existingName = RetrieveVariableNameOrCreateOne(colName);
            // Declare name in IR
            var name = CreateVariableName(colName);
            // Create TypeProto according to the input column
            var typeProto = PmfUtils.MakeType(columnType);
            // Create ParameterDeclProto based on information above
            _modelOutputParameters[name] = typeProto;
            _refPool[name] = PmfUtils.MakeVarRef(name);
            AddExpression(PmfUtils.MakeSet(name, _refPool[existingName]));
        }
        public void AddExpression(LotusvNext.Expressions.Expression expression)
        {
            _expressions.Add(expression);
        }
        // Review: Make... should be moved to PmfContext
        public FunctionalModelProto MakeFunctionalModel()
        {
            var inputs = new List<LotusvNext.Types.TypeProto.Types.ParameterDeclProto>();
            var outputs = new List<LotusvNext.Types.TypeProto.Types.ParameterDeclProto>();
            foreach (var i in _modelInputParameters)
                inputs.Add(PmfUtils.MakeParam(i.Key, i.Value));
            foreach (var i in _modelOutputParameters)
                outputs.Add(PmfUtils.MakeParam(i.Key, i.Value));

            var signature = PmfUtils.MakeSignature(inputs, outputs);

            return PmfUtils.MakeFunctionalModel("Model", signature, _expressions, _functions, _varTypes);
        }
        public ModelProto MakeModel()
        {
            var modelProto = new ModelProto
            {
                Profile = "ONNX",
                IrVersion = 3L,
                ProducerName = "ML.NET",
                ProducerVersion = "0",
                Domain = "com.microsoft",
                ModelVersion = 0L,
                Model = MakeFunctionalModel()
            };
            return modelProto;
        }

        public Expression GetRef(string name)
        {
            return _refPool[name];
        }
        public Expression GetDef(string name)
        {
            return _defPool[name];
        }
        public Expression GetExp(string name)
        {
            if (_defPool[name].Let.VariableBindings.Count() != 1)
                return null;
            return _defPool[name].Let.VariableBindings[0].InitialValue;
        }

        // Int64 scalar
        public string Declare(long val, string name=null)
        {
            var exps = new List<Expression>();

            if (name == null)
                name = CreateVariableName("int64Value");

            var valExp = PmfUtils.Make(val);
            _defPool[name] = _modelOutputParameters.ContainsKey(name) ? PmfUtils.MakeSet(name, valExp) : PmfUtils.MakeLet(name, valExp);

            var valRef = PmfUtils.MakeVarRef(name);
            _refPool[name] = valRef;

            return name;
        }
        // Int64 array (1-D tensor)
        public string Declare(IEnumerable<long> vals, string name=null)
        {
            var exps = new List<Expression>();

            if (name == null)
                name = CreateVariableName("int64Array");

            var valExp = PmfUtils.Make(vals);
            _defPool[name] = _modelOutputParameters.ContainsKey(name) ? PmfUtils.MakeSet(name, valExp) : PmfUtils.MakeLet(name, valExp);

            var valRef = PmfUtils.MakeVarRef(name);
            _refPool[name] = valRef;

            return name;
        }
        // Float
        public string Declare(float val, string name=null)
        {
            var exps = new List<Expression>();

            if (name == null)
                name = CreateVariableName("floatValue");

            var valExp = PmfUtils.Make(val);
            _defPool[name] = _modelOutputParameters.ContainsKey(name) ? PmfUtils.MakeSet(name, valExp) : PmfUtils.MakeLet(name, valExp);

            var valRef = PmfUtils.MakeVarRef(name);
            _refPool[name] = valRef;

            return name;
        }
        // Float array (1-D tensor)
        public string Declare(IEnumerable<float> vals, string name=null)
        {
            var exps = new List<Expression>();

            if (name == null)
                name = CreateVariableName("floatArray");

            var valExp = PmfUtils.Make(vals);
            _defPool[name] = _modelOutputParameters.ContainsKey(name) ? PmfUtils.MakeSet(name, valExp) : PmfUtils.MakeLet(name, valExp);

            var valRef = PmfUtils.MakeVarRef(name);
            _refPool[name] = valRef;

            return name;
        }
        // String
        public string Declare(string val, string name=null)
        {
            var exps = new List<Expression>();

            if (name == null)
                name = CreateVariableName("stringValue");

            var valExp = PmfUtils.Make(val);
            _defPool[name] = _modelOutputParameters.ContainsKey(name) ? PmfUtils.MakeSet(name, valExp) : PmfUtils.MakeLet(name, valExp);

            var valRef = PmfUtils.MakeVarRef(name);
            _refPool[name] = valRef;

            return name;
        }
        // String (1-D tensor)
        public string Declare(IEnumerable<string> vals, string name=null)
        {
            var exps = new List<Expression>();

            if (name == null)
                name = CreateVariableName("stringArray");

            var valExp = PmfUtils.Make(vals);
            _defPool[name] = _modelOutputParameters.ContainsKey(name) ? PmfUtils.MakeSet(name, valExp) : PmfUtils.MakeLet(name, valExp);

            var valRef = PmfUtils.MakeVarRef(name);
            _refPool[name] = valRef;

            return name;
        }
        public string Declare(ColumnType colType, string name=null)
        {
            var typeProto = PmfUtils.MakeType(colType);

            if (name == null)
                name = CreateVariableName("column");

            var valExp = PmfUtils.MakeDefault(colType);
            _defPool[name] = _modelOutputParameters.ContainsKey(name) ? PmfUtils.MakeSet(name, valExp) : PmfUtils.MakeLet(name, valExp);

            var valRef = PmfUtils.MakeVarRef(name);
            _refPool[name] = valRef;

            _varTypes[name] = typeProto;

            return name;
        }
        // Indexing
        public string Access(LotusvNext.Expressions.Expression container, LotusvNext.Expressions.Expression path, string name=null)
        {
            if (name == null)
                name = CreateVariableName("column");

            var accessExp = PmfUtils.MakeElementAccess(container, path);
            var accessDef = PmfUtils.MakeLet(name, accessExp);
            _defPool[name] = accessDef;

            var accessRef = PmfUtils.MakeVarRef(name);
            _refPool[name] = accessRef;

            return name;
        }
        // For
        public LotusvNext.Expressions.Expression MakeFor(
            LotusvNext.Expressions.Let.Types.Binding induction,
            LotusvNext.Expressions.Expression condition,
            LotusvNext.Expressions.Set step,
            IEnumerable<LotusvNext.Expressions.Expression> body=null)
        {
            return new LotusvNext.Expressions.Expression()
            {
                For = PmfUtils.MakeForProto(induction, condition, step, body)
            };
        }
        public LotusvNext.Expressions.Expression MakeFor(
            string iName, long iStart, long iEnd, long iStep=1)
        {
            var binding = PmfUtils.MakeBinding(iName, PmfUtils.Make(iStart));
            var iEndExp = PmfUtils.Make(iEnd);
            var condExp = PmfUtils.Call("Less", GetRef(iName), iEndExp);
            var iStepProto = PmfUtils.MakeSet(iName, PmfUtils.Make(iStep)).Set;

            return MakeFor(binding, condExp, iStepProto);
        }
        public LotusvNext.Expressions.Expression MakeFor(
            string iName, string startName, string endName, long iStep=1)
        {
            var binding = PmfUtils.MakeBinding(iName, PmfUtils.MakeVarRef(startName));
            var cond = PmfUtils.Call("Less", GetRef(iName) , PmfUtils.MakeVarRef(endName));
            var iStepProto = PmfUtils.MakeSet(iName, PmfUtils.Make(iStep)).Set;

            return MakeFor(binding, cond, iStepProto);
        }
        public LotusvNext.Expressions.Expression MakeForEach(
            string iName, LotusvNext.Expressions.Expression collection)
        {
            var forEachProto = new LotusvNext.Expressions.ForEach()
            {
                Variable = iName,
                Sequence = collection
            };
            var forEachExp = new LotusvNext.Expressions.Expression()
            {
                ForEach = forEachProto
            };
            return forEachExp;
        }
    }
}
