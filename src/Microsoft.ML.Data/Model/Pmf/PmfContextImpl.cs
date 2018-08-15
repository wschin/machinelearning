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
        // Each (key, value) pair is a pair from ML.NET column name to the associated variable name in IR
        private Dictionary<string, string> _nameMap;
        // All existing variable names. New variables must not exist in this set.
        // All existing node names. New node names must not alrady exist in this set.
        private List<LotusvNext.Types.TypeProto.Types.ParameterDeclProto> _modelInputs;
        private List<LotusvNext.Types.TypeProto.Types.ParameterDeclProto> _modelOutputs;
        private Dictionary<string, LotusvNext.Types.TypeProto> _varTypes;
        private Dictionary<string, LotusvNext.Expressions.Expression> _refPool;
        private Dictionary<string, LotusvNext.Expressions.Expression> _defPool;
        Dictionary<string, LotusvNext.FunctionDefProto> _functions;
        private List<LotusvNext.Expressions.Expression> _expressions;

        public PmfContext()
        {
            _nameMap = new Dictionary<string, string>();
            _modelInputs = new List<LotusvNext.Types.TypeProto.Types.ParameterDeclProto>();
            _modelOutputs = new List<LotusvNext.Types.TypeProto.Types.ParameterDeclProto>();
            _varTypes = new Dictionary<string, LotusvNext.Types.TypeProto>();
            _functions = new Dictionary<string, LotusvNext.FunctionDefProto>();
            _expressions = new List<LotusvNext.Expressions.Expression>();
            _refPool = new Dictionary<string, Expression>();
            _defPool = new Dictionary<string, Expression>();
        }

        private string CreateUniqueString(ref Dictionary<string, string> pool, string prefix)
        {
            if (!pool.ContainsValue(prefix))
            {
                pool.Add(prefix, prefix);
                return prefix;
            }

            // The index will be appended
            int count = 1;
            // Candidate string
            string derivedString = prefix + count;;
            // Keep increasing count until prefix + count cannot be found in the pool
            while(pool.ContainsValue(derivedString))
            {
                derivedString = prefix + ++count;
            }
            // Add derived_string into the pool so that we will not declare the same string again in the future
            pool[prefix] = derivedString;
            return derivedString;
        }

        public string DeclareRef(string prefix)
        {
            string name = CreateUniqueString(ref _nameMap, prefix);
            _nameMap[prefix] = name;
            _refPool[name] = PmfUtils.MakeVarRef(name);
            return name;
        }

        public string Retrieve(string prefix)
        {
            return _nameMap[prefix];
        }

        // Use to create I/O for FunctionalModelProto
        public void AddModelInput(ColumnType columnType, string colName)
        {
            var name = DeclareRef(colName);
            var typeProto = PmfUtils.MakeType(columnType);
            _modelInputs.Add(PmfUtils.MakeParam(name, typeProto));
        }

        public void AddModelOutput(ColumnType columnType, string colName)
        {
            var intermediateName = Retrieve(colName);

            var name = DeclareRef(colName);
            var typeProto = PmfUtils.MakeType(columnType);
            _modelOutputs.Add(PmfUtils.MakeParam(name, typeProto));

            _defPool[name] = PmfUtils.MakeSet(name, _refPool[intermediateName]);
            AddExp(_defPool[name]);
        }
        public void AddExp(params LotusvNext.Expressions.Expression[] expression)
        {
            _expressions.AddRange(expression);
        }
        // Review: Make... should be moved to PmfContext
        public FunctionalModelProto MakeFunctionalModel()
        {
            var signature = PmfUtils.MakeSignature(_modelInputs, _modelOutputs);
            return PmfUtils.MakeFunctionalModel("Model", signature, _expressions, _functions, _varTypes);
        }
        public ModelProto MakeModel()
        {
            return PmfUtils.MakeModel(MakeFunctionalModel());
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
                name = DeclareRef("int64Value");

            var valExp = PmfUtils.Make(val);
            _defPool[name] = PmfUtils.MakeLet(name, valExp);

            var valRef = PmfUtils.MakeVarRef(name);
            _refPool[name] = valRef;

            return name;
        }
        // Int64 array (1-D tensor)
        public string Declare(IEnumerable<long> vals, string name=null)
        {
            var exps = new List<Expression>();

            if (name == null)
                name = DeclareRef("int64Array");

            var valExp = PmfUtils.Make(vals);
            _defPool[name] = PmfUtils.MakeLet(name, valExp);

            var valRef = PmfUtils.MakeVarRef(name);
            _refPool[name] = valRef;

            return name;
        }
        // Float
        public string Declare(float val, string name=null)
        {
            var exps = new List<Expression>();

            if (name == null)
                name = DeclareRef("floatValue");

            var valExp = PmfUtils.Make(val);
            _defPool[name] = PmfUtils.MakeLet(name, valExp);

            var valRef = PmfUtils.MakeVarRef(name);
            _refPool[name] = valRef;

            return name;
        }
        // Float array (1-D tensor)
        public string Declare(IEnumerable<float> vals, string name=null)
        {
            var exps = new List<Expression>();

            if (name == null)
                name = DeclareRef("floatArray");

            var valExp = PmfUtils.Make(vals);
            _defPool[name] = PmfUtils.MakeLet(name, valExp);

            var valRef = PmfUtils.MakeVarRef(name);
            _refPool[name] = valRef;

            return name;
        }
        // String
        public string Declare(string val, string name=null)
        {
            var exps = new List<Expression>();

            if (name == null)
                name = DeclareRef("stringValue");

            var valExp = PmfUtils.Make(val);
            _defPool[name] = PmfUtils.MakeLet(name, valExp);

            var valRef = PmfUtils.MakeVarRef(name);
            _refPool[name] = valRef;

            return name;
        }
        // String (1-D tensor)
        public string Declare(IEnumerable<string> vals, string name=null)
        {
            var exps = new List<Expression>();

            if (name == null)
                name = DeclareRef("stringArray");

            var valExp = PmfUtils.Make(vals);
            _defPool[name] = PmfUtils.MakeLet(name, valExp);

            var valRef = PmfUtils.MakeVarRef(name);
            _refPool[name] = valRef;

            return name;
        }
        public string Declare(ColumnType colType, string name=null)
        {
            var typeProto = PmfUtils.MakeType(colType);

            if (name == null)
                name = DeclareRef("column");

            var valExp = PmfUtils.MakeDefault(colType);
            _defPool[name] = PmfUtils.MakeLet(name, valExp);

            var valRef = PmfUtils.MakeVarRef(name);
            _refPool[name] = valRef;

            _varTypes[name] = typeProto;

            return name;
        }
        // Indexing
        public string Access(LotusvNext.Expressions.Expression container, LotusvNext.Expressions.Expression path, string name=null)
        {
            if (name == null)
                name = DeclareRef("column");

            var accessExp = PmfUtils.MakeElementAccess(container, path);
            var accessDef = PmfUtils.MakeLet(name, accessExp);
            _defPool[name] = accessDef;

            var accessRef = PmfUtils.MakeVarRef(name);
            _refPool[name] = accessRef;

            return name;
        }
    }
}
