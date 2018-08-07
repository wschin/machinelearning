// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Runtime.Model.Pmf
{
    public sealed class PmfUtils
    {
        // Functions for making ValueProto are defined below
        public static LotusvNext.Types.TypeProto.Types.TensorTypeProto MakeTensorTypeProto(
            ONNX.TensorProto.Types.DataType elemType, List<long> dims)
        {
            // First we make a TensorTypeProto because its an one-of field in TypeProto
            var tensorTypeProto = new LotusvNext.Types.TypeProto.Types.TensorTypeProto();
            // Directly assign element type because it's a primitive protobuf type
            tensorTypeProto.ElemType = elemType;
            // The shape information is a customized data structure, we need to allocate one before modifying it
            tensorTypeProto.Shape = new ONNX.TensorShapeProto();

            // Scan through the input dimension list and copy the stored information into ONNX's shape object
            for (int i = 0; i < dims.Count; ++i)
            {
                var d = new ONNX.TensorShapeProto.Types.Dimension();
                if (dims[i] < 1)  // zero or negative dimension means variable-length
                    d.DimParam = "None";
                else
                    d.DimValue = dims[i]; // positive integer provides dimension explicitly 
                tensorTypeProto.Shape.Dim.Add(d);
            }
            return tensorTypeProto;
        }
        public static LotusvNext.Types.TypeProto MakeTypeProtoTensor(
            ONNX.TensorProto.Types.DataType elemType, List<long> dims)
        {
            // TypeProto and then assign the TensorTypeProto to it.
            var typeProto = new LotusvNext.Types.TypeProto();
            typeProto.TensorType = MakeTensorTypeProto(elemType, dims);
            return typeProto;
        }
        public static LotusvNext.Types.ValueInfoProto MakeValueInfoProtoTensor(
            string name, ONNX.TensorProto.Types.DataType elemType,
            List<long> dims, string docString="")
        {
            var valueInfo = new LotusvNext.Types.ValueInfoProto
            {
                Name = name,
                Type = MakeTypeProtoTensor(elemType, dims),
                DocString = docString
            };
            return valueInfo;
        }

        public static LotusvNext.Types.TypeProto.Types.ParameterDeclProto MakeParameterDeclProtoTensor(
            string name, LotusvNext.Types.TypeProto type,
            string docString="", bool variadic=false)
        {
            var parameterDeclProto = new LotusvNext.Types.TypeProto.Types.ParameterDeclProto
            {
                Name = name,
                Type = type,
                DocString = docString,
                Variadic = variadic
            };
            return parameterDeclProto;
        }

        public static LotusvNext.Types.TypeProto.Types.SignatureDeclProto MakeSignatureDeclProtoTensor(
            List<LotusvNext.Types.TypeProto.Types.ParameterDeclProto> inputParams,
            List<LotusvNext.Types.TypeProto.Types.ParameterDeclProto> outputParams,
            //List<LotusvNext.Types.TypeProto.Types.ParameterDeclProto> attributeParams,
            string docString)
        {
            var signatureDeclProto = new LotusvNext.Types.TypeProto.Types.SignatureDeclProto();
            signatureDeclProto.InputParams.AddRange(inputParams);
            signatureDeclProto.OutputParams.AddRange(outputParams);
            //signatureDeclProto.InputAttributes.AddRange(attributeParams);
            signatureDeclProto.DocString = docString;
            return signatureDeclProto;
        }
        
        public static LotusvNext.Expressions.Let.Types.Binding MakeBinding(
            string name, LotusvNext.Expressions.Expression expression)
        {
            var binding = new LotusvNext.Expressions.Let.Types.Binding
            {
                Name = name,
                InitialValue = expression
            };
            return binding;
        }
        public static LotusvNext.Types.TypeProto MakeTypeProtoScalar(
            ONNX.TensorProto.Types.DataType type)
        {
            var typeProto = new LotusvNext.Types.TypeProto
            {
                ScalarType = type
            };
            return typeProto;
        }

        public static LotusvNext.Expressions.ValueProto MakeValueProtoInt64Scalar(long value)
        {
            var valueProto = new LotusvNext.Expressions.ValueProto
            {
                Integer = value,
                Type = MakeTypeProtoScalar(ONNX.TensorProto.Types.DataType.Int64)
            };
            return valueProto;
        }

        public static LotusvNext.Expressions.ValueProto MakeValueProtoFloatScalar(float value)
        {
            var valueProto = new LotusvNext.Expressions.ValueProto
            {
                Float = value,
                Type = MakeTypeProtoScalar(ONNX.TensorProto.Types.DataType.Float)
            };
            return valueProto;
        }
        public static ONNX.TensorProto MakeTensorProtoFloat(
            string name, List<long> dims, List<float> values, string docString)
        {
            var tensorProto = new ONNX.TensorProto();
            tensorProto.Name = name;
            tensorProto.Dims.AddRange(dims);
            tensorProto.DocString = docString;
            return tensorProto;
        }
        public static LotusvNext.Expressions.ValueProto MakeValueProtoFloatTensor(
            string name, List<long> dims, List<float> values, string docString="")
        {
            var valueProto = new LotusvNext.Expressions.ValueProto()
            {
                DenseTensor = MakeTensorProtoFloat(name, dims, values, docString)
            };
            return valueProto;
        }
        public static LotusvNext.Expressions.ValueProto.Types.KeyValuePairProto MakeKeyValuePairProtoInt64ScalarToFloatScalar(
            long key, float value)
        {
            var keyValuePairProto = new LotusvNext.Expressions.ValueProto.Types.KeyValuePairProto()
            {
                I64 = key,
                Value = MakeValueProtoFloatScalar(value)
            };
            return keyValuePairProto;
        }
        public static LotusvNext.Expressions.ValueProto.Types.MapProto MakeMapProtoInt64ScalarToFloatScalar(
            Dictionary<long, float> dictionary)
        {
            var mapProto = new LotusvNext.Expressions.ValueProto.Types.MapProto();
            mapProto.KeyValuePairs.AddRange(dictionary.Select(x => MakeKeyValuePairProtoInt64ScalarToFloatScalar(x.Key, x.Value)));
            return mapProto;
        }
        public static LotusvNext.Expressions.ValueProto MakeValueProtoMapInt64ScalarToFloatScalar(
            Dictionary<long, float> dictionary)
        {
            var valueProto = new LotusvNext.Expressions.ValueProto()
            {
                Map = MakeMapProtoInt64ScalarToFloatScalar(dictionary)
            };
            return valueProto;
        }
 
        // Functions for making expressions are defined below
        public static LotusvNext.Expressions.VariableReference MakeVariableReferenceProto(string name)
        {
            var variableReferenceProto = new LotusvNext.Expressions.VariableReference()
            {
                Name = name
            };
            return variableReferenceProto;
        }
        public static LotusvNext.Expressions.Expression MakeValueProtoVariableReference(string name)
        {
            var valueProto = new LotusvNext.Expressions.Expression()
            {
                Variable = MakeVariableReferenceProto(name)
            };
            return valueProto;
        }
        public static LotusvNext.Expressions.Expression MakeElementAccess(
            LotusvNext.Expressions.Expression container,
            List<LotusvNext.Expressions.Expression> path,
            LotusvNext.Expressions.Expression defaultValue=null)
        {
            var elementAccess = new LotusvNext.Expressions.ElementAccess();
            elementAccess.Container = container;
            elementAccess.Path.AddRange(path);
            if (defaultValue != null)
                elementAccess.Default = defaultValue;
            var expression = new LotusvNext.Expressions.Expression()
            {
                Index = elementAccess
            };
            return expression;
        }
        public static LotusvNext.Expressions.Expression MakeElementAccess(
            LotusvNext.Expressions.Expression container,
            LotusvNext.Expressions.Expression path,
            LotusvNext.Expressions.Expression defaultValue=null)
        {
            var elementAccess = new LotusvNext.Expressions.ElementAccess();
            elementAccess.Container = container;
            elementAccess.Path.Add(path);
            if (defaultValue != null)
                elementAccess.Default = defaultValue;
            var expression = new LotusvNext.Expressions.Expression()
            {
                Index = elementAccess
            };
            return expression;
        }

        public static LotusvNext.Expressions.Let.Types.Binding MakeBindingProto(string name, LotusvNext.Expressions.Expression initialValue)
        {
            var bindingProto = new LotusvNext.Expressions.Let.Types.Binding()
            {
                Name = name,
                InitialValue = initialValue
            };
            return bindingProto;
        }
        public static LotusvNext.Expressions.Let MakeLetProto(List<LotusvNext.Expressions.Let.Types.Binding> bindings)
        {
            var letProto = new LotusvNext.Expressions.Let();
            letProto.VariableBindings.AddRange(bindings);
            return letProto;
        }
        public static LotusvNext.Expressions.Let MakeLetProto(LotusvNext.Expressions.Let.Types.Binding binding)
        {
            var letProto = new LotusvNext.Expressions.Let();
            letProto.VariableBindings.Add(binding);
            return letProto;
        }
        public static LotusvNext.Expressions.Expression MakeLetExpression(List<LotusvNext.Expressions.Let.Types.Binding> bindings)
        {
            return new LotusvNext.Expressions.Expression()
            {
                Let = MakeLetProto(bindings)
            };
        }
        public static LotusvNext.Expressions.Expression MakeLetExpression(LotusvNext.Expressions.Let.Types.Binding bindings)
        {
            return new LotusvNext.Expressions.Expression()
            {
                Let = MakeLetProto(bindings)
            };
        }
        public static LotusvNext.Expressions.Set MakeSetProto(string name, LotusvNext.Expressions.Expression value)
        {
            var setProto = new LotusvNext.Expressions.Set()
            {
                Name = name,
                Value = value
            };
            return setProto;
        }
        public static LotusvNext.Expressions.Expression MakeSet(string name, LotusvNext.Expressions.Expression exp)
        {
            return new LotusvNext.Expressions.Expression()
            {
                Set=MakeSetProto(name, exp)
            };
        }
        public static LotusvNext.Expressions.For MakeForProto(
            List<LotusvNext.Expressions.Let.Types.Binding> inductionVariables,
            LotusvNext.Expressions.Expression condition,
            List<LotusvNext.Expressions.Expression> body,
            List<LotusvNext.Expressions.Set> step)
        {
            var forProto = new LotusvNext.Expressions.For();
            forProto.InductionVariables.AddRange(inductionVariables);
            forProto.Condition = condition;
            forProto.Body.AddRange(body);
            forProto.Step.AddRange(step);
            return forProto;
        }
        public static LotusvNext.Expressions.Expression MakeFor(
            LotusvNext.Expressions.Let.Types.Binding induction,
            LotusvNext.Expressions.Expression condition,
            List<LotusvNext.Expressions.Expression> body,
            LotusvNext.Expressions.Set step)
        {
            return new LotusvNext.Expressions.Expression()
            {
                For = MakeForProto(new List<LotusvNext.Expressions.Let.Types.Binding>() { induction },
                condition, body, new List<LotusvNext.Expressions.Set>() { step })
            };
        }
        public static LotusvNext.Expressions.FunctionReference MakeFunctionReferenceProto(string name)
        {
            var functionReferenceProto = new LotusvNext.Expressions.FunctionReference()
            { Name = name };
            return functionReferenceProto;
        }
        public static LotusvNext.Expressions.Expression MakeFunctionReferenceExpression(string name)
        {
            return new LotusvNext.Expressions.Expression()
            {
                Funcref = MakeFunctionReferenceProto(name)
            };
        }
        public static LotusvNext.Expressions.Call MakeCallProto(
            LotusvNext.Expressions.Expression target,
            List<LotusvNext.Expressions.Expression> positional)
        {
            var callProto = new LotusvNext.Expressions.Call();
            callProto.Target = target; // this is usually a function reference
            callProto.Positional.AddRange(positional);
            return callProto;
        }
        public static LotusvNext.Expressions.Expression MakeCall(
            LotusvNext.Expressions.Expression target,
            List<LotusvNext.Expressions.Expression> positional)
        {
            return new LotusvNext.Expressions.Expression()
            {
                Call = MakeCallProto(target, positional)
            };
        }
        public static LotusvNext.Expressions.Expression MakeCall(
            LotusvNext.Expressions.Expression target,
            LotusvNext.Expressions.Expression left,
            LotusvNext.Expressions.Expression right)
        {
            return new LotusvNext.Expressions.Expression()
            {
                Call = MakeCallProto(target, new List<LotusvNext.Expressions.Expression>() { left, right })
            };
        }
        
        public static LotusvNext.Expressions.Expression MakeInt64LiteralExpression(long value)
        {
            return new LotusvNext.Expressions.Expression()
            {
                Literal = MakeValueProtoInt64Scalar(value)
            };
        }
        public static LotusvNext.Expressions.Expression MakeFloatLiteralExpression(float value)
        {
            return new LotusvNext.Expressions.Expression()
            {
                Literal = MakeValueProtoFloatScalar(value)
            };
        }

        public static LotusvNext.OperatorSetIdProto MakeOperatorSetIdProto(string domain, long version)
        {
            return new LotusvNext.OperatorSetIdProto()
            {
                Domain = domain,
                Version = version
            };
        }
        public static LotusvNext.FunctionalModelProto MakeFunctionalModelProto(
            string name,
            Dictionary<string, LotusvNext.Types.TypeProto> types,
            Dictionary<string, LotusvNext.FunctionDefProto> functions,
            List<LotusvNext.Expressions.Expression> body
            )
        {
            var functionalModelProto = new LotusvNext.FunctionalModelProto();
            functionalModelProto.Name = name;
            functionalModelProto.Types_.Add(types);
            functionalModelProto.Body.AddRange(body);
            return functionalModelProto;
        }
        public static LotusvNext.ModelProto MakeModelProto(LotusvNext.FunctionalModelProto model)
        {
            var modelProto = new LotusvNext.ModelProto()
            {
                IrVersion = 0,
                ProducerName = "ML.NET",
                ProducerVersion = "0",
                Domain = "ML.NET",
                ModelVersion = 0,
                Model=model
                // Should type be specified?
            };
            return modelProto;
        }
        public static LotusvNext.Types.TypeProto TranslateColumnTypeToTypeProto(ColumnType type)
        {
            var typeProto = new LotusvNext.Types.TypeProto();
            ONNX.TensorProto.Types.DataType typeId = ONNX.TensorProto.Types.DataType.Undefined;
            DataKind rawKind;
            if (type.IsVector)
                rawKind = type.AsVector.ItemType.RawKind;
            else if (type.IsKey)
                rawKind = type.AsKey.RawKind;
            else
                rawKind = type.RawKind;

            switch (rawKind)
            {
                case DataKind.BL:
                    typeId = ONNX.TensorProto.Types.DataType.Float;
                    break;
                case DataKind.TX:
                    typeId = ONNX.TensorProto.Types.DataType.String;
                    break;
                case DataKind.U4:
                    typeId = ONNX.TensorProto.Types.DataType.Int64;
                    break;
                case DataKind.R4:
                    typeId = ONNX.TensorProto.Types.DataType.Float;
                    break;
                default:
                    Contracts.Assert(false, "Unknown type.");
                    break;
            }

            if (type.IsVector)
            {
                // Return a TypeProto with tensor_type activated
                var dims = new List<long>();
                if (type.ValueCount == 0) //Unknown size.
                    dims.Add(-1);
                else if (type.ValueCount == 1)
                    dims.Add(1);
                else if (type.ValueCount > 1)
                {
                    var vec = type.AsVector;
                    for (int i = 0; i < vec.DimCount; i++)
                        dims.Add(vec.GetDim(i));
                }
                // Batch size, which is the first element in the dimension list
                dims?.Insert(0, 1);
                return MakeTypeProtoTensor(typeId, dims);
            }
            else
            {
                // Assumes that only scalar is possible. Not sure if other types such as dictionary will present. 
                return MakeTypeProtoScalar(typeId);
            }
        }

        public static LotusvNext.Expressions.Expression MakeFloatTensorValue(
            string name, List<long> dims, float[] floats)
        {
            var tensorProto = new ONNX.TensorProto();
            for (int i = 0; i < dims.Count; ++i)
                tensorProto.Dims.Add(dims[i]);
            tensorProto.FloatData.AddRange(floats);

            var valueProto = new LotusvNext.Expressions.ValueProto()
            {
                DenseTensor = tensorProto
            };

            var expressionProto = new LotusvNext.Expressions.Expression()
            {
                Literal = valueProto
            };

            return expressionProto;
        }

        public static LotusvNext.FunctionalModelProto MakeFunctionalModelProto(
            string name,
            List<LotusvNext.Types.TypeProto.Types.ParameterDeclProto> inputs,
            List<LotusvNext.Types.TypeProto.Types.ParameterDeclProto> outputs,
            List<LotusvNext.Expressions.Expression> body)
        {
            var signature = PmfUtils.MakeSignatureDeclProtoTensor(inputs, outputs, "Function inputs and outputs");
            var model = new LotusvNext.FunctionalModelProto();
            model.Name = name;
            model.Signature = signature;
            model.Body.AddRange(body);
            System.Console.Write(model);
            return model;
        }
    }
    /// <summary>
    /// A context for defining a ONNX output. The context internally contains the model-in-progress being built. This
    /// same context object is iteratively given to exportable components via the <see cref="ICanSavePmf"/> interface
    /// and subinterfaces, that attempt to express their operations as ONNX nodes, if they can. At the point that it is
    /// given to a component, all other components up to that component have already attempted to express themselves in
    /// this context, with their outputs possibly available in the ONNX graph.
    /// </summary>
    public abstract class PmfContext
    {
        public abstract string CreateOperatorName(string prefix);
        public abstract string CreateVariableName(string prefix);
        public abstract string RetrieveVariableNameOrCreateOne(string prefix);
        public abstract void AddInputVariable(ColumnType columnType, string colName);
        public abstract void AddOutputVariable(ColumnType columnType, string variableName);
        public abstract void AddExpression(LotusvNext.Expressions.Expression expression);
        public abstract LotusvNext.FunctionalModelProto MakeFunctionalModel();
        public abstract LotusvNext.ModelProto MakeModel();
    }
}