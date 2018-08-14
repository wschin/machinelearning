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
        // Helper functions for creating values 
        public static LotusvNext.Expressions.Expression Make(long value)
        {
            var valueProto = new LotusvNext.Expressions.ValueProto
            {
                Integer = value,
                Type = MakeInt64Type()
            };

            var expressionProto = new LotusvNext.Expressions.Expression()
            {
                Literal = valueProto
            };

            return expressionProto;
        }
        public static LotusvNext.Expressions.Expression Make(IEnumerable<long> values)
        {
            var tensorProto = new ONNX.TensorProto();
            tensorProto.DataType = ONNX.TensorProto.Types.DataType.Int64;
            tensorProto.Dims.Add(values.LongCount());
            tensorProto.Int64Data.AddRange(values);

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
        public static LotusvNext.Expressions.Expression Make(IEnumerable<long> values, IEnumerable<long> dims)
        {
            var tensorProto = new ONNX.TensorProto();
            tensorProto.DataType = ONNX.TensorProto.Types.DataType.Int64;
            tensorProto.Dims.AddRange(dims);
            tensorProto.Int64Data.AddRange(values);

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
        public static LotusvNext.Expressions.Expression Make(float value)
        {
            var valueProto = new LotusvNext.Expressions.ValueProto
            {
                Float = value, // this field's type is double, we should add float field into ValueProto.
                Type = MakeFloatType()
            };

            var expressionProto = new LotusvNext.Expressions.Expression()
            {
                Literal = valueProto
            };

            return expressionProto;
        }
        public static LotusvNext.Expressions.Expression Make(IEnumerable<float> values)
        {
            var tensorProto = new ONNX.TensorProto();
            tensorProto.DataType = ONNX.TensorProto.Types.DataType.Float;
            tensorProto.Dims.Add(values.LongCount());
            tensorProto.FloatData.AddRange(values);

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
        public static LotusvNext.Expressions.Expression Make(IEnumerable<float> values, IEnumerable<long> dims)
        {
            var tensorProto = new ONNX.TensorProto();
            tensorProto.DataType = ONNX.TensorProto.Types.DataType.Float;
            tensorProto.Dims.AddRange(dims);
            tensorProto.FloatData.AddRange(values);

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
        public static LotusvNext.Expressions.Expression Make(string value)
        {
            var valueProto = new LotusvNext.Expressions.ValueProto
            {
                String = value, // this field's type is double, we should add float field into ValueProto.
                Type = MakeStringType()
            };

            var expressionProto = new LotusvNext.Expressions.Expression()
            {
                Literal = valueProto
            };

            return expressionProto;
        }
        public static LotusvNext.Expressions.Expression Make(IEnumerable<string> values)
        {
            var tensorProto = new ONNX.TensorProto();
            tensorProto.DataType = ONNX.TensorProto.Types.DataType.String;
            tensorProto.Dims.Add(values.LongCount());
            tensorProto.StringData.AddRange(values.Select(s => Google.Protobuf.ByteString.CopyFrom(System.Text.Encoding.UTF8.GetBytes(s))));

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
        public static LotusvNext.Expressions.Expression Make(IEnumerable<string> values, IEnumerable<long> dims)
        {
            var tensorProto = new ONNX.TensorProto();
            tensorProto.DataType = ONNX.TensorProto.Types.DataType.String;
            tensorProto.Dims.AddRange(dims);
            tensorProto.StringData.AddRange(values.Select(s => Google.Protobuf.ByteString.CopyFrom(System.Text.Encoding.UTF8.GetBytes(s))));

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
        public static LotusvNext.Expressions.ValueProto.Types.KeyValuePairProto MakePair(string key, long value)
        {
            return new LotusvNext.Expressions.ValueProto.Types.KeyValuePairProto()
            {
                S = key,
                Value = Make(value).Literal
            };
        }
        public static LotusvNext.Expressions.ValueProto.Types.KeyValuePairProto MakePair(long key, string value)
        {
            return new LotusvNext.Expressions.ValueProto.Types.KeyValuePairProto()
            {
                I64 = key,
                Value = Make(value).Literal
            };
        }
        public static LotusvNext.Expressions.ValueProto.Types.KeyValuePairProto MakePair(string key, string value)
        {
            return new LotusvNext.Expressions.ValueProto.Types.KeyValuePairProto()
            {
                S= key,
                Value = Make(value).Literal
            };
        }
        public static LotusvNext.Expressions.ValueProto.Types.KeyValuePairProto MakePair(long key, long value)
        {
            return new LotusvNext.Expressions.ValueProto.Types.KeyValuePairProto()
            {
                I64= key,
                Value = Make(value).Literal
            };
        }
        public static LotusvNext.Expressions.ValueProto.Types.KeyValuePairProto MakePair(long key, float value)
        {
            return new LotusvNext.Expressions.ValueProto.Types.KeyValuePairProto()
            {
                I64= key,
                Value = Make(value).Literal
            };
        }
        public static LotusvNext.Expressions.ValueProto.Types.KeyValuePairProto MakePair(string key, float value)
        {
            return new LotusvNext.Expressions.ValueProto.Types.KeyValuePairProto()
            {
                S = key,
                Value = Make(value).Literal
            };
        }
        public static LotusvNext.Expressions.Expression MakeDictionary(IEnumerable<string> keys=null, IEnumerable<long> values=null)
        {
            var mapProto = new LotusvNext.Expressions.ValueProto.Types.MapProto();
            if (keys != null && values != null)
            {
                using (var ik = keys.GetEnumerator())
                using (var iv = values.GetEnumerator())
                    while (ik.MoveNext() && iv.MoveNext())
                    {
                        if (values != null)
                            mapProto.KeyValuePairs.Add(MakePair(ik.Current, iv.Current));
                        else
                            mapProto.KeyValuePairs.Add(MakePair(ik.Current, default(long)));
                    }
            }

            var valueProto = new LotusvNext.Expressions.ValueProto();
            valueProto.Map = mapProto;

            var expProto = new LotusvNext.Expressions.Expression();
            expProto.Literal = valueProto;
            return expProto;
        }
        public static LotusvNext.Expressions.Expression MakeDictionary(IEnumerable<long> keys=null, IEnumerable<string> values=null)
        {
            var mapProto = new LotusvNext.Expressions.ValueProto.Types.MapProto();
            if (keys != null && values != null)
            {
                using (var ik = keys.GetEnumerator())
                using (var iv = values.GetEnumerator())
                    while (ik.MoveNext() && iv.MoveNext())
                    {
                        if (values != null)
                            mapProto.KeyValuePairs.Add(MakePair(ik.Current, iv.Current));
                        else
                            mapProto.KeyValuePairs.Add(MakePair(ik.Current, default(string)));
                    }
            }

            var valueProto = new LotusvNext.Expressions.ValueProto();
            valueProto.Map = mapProto;

            var expProto = new LotusvNext.Expressions.Expression();
            expProto.Literal = valueProto;
            return expProto;
        }
        public static LotusvNext.Expressions.Expression MakeDictionary(IEnumerable<string> keys=null, IEnumerable<string> values=null)
        {
            var mapProto = new LotusvNext.Expressions.ValueProto.Types.MapProto();
            if (keys != null && values != null)
            {
                using (var ik = keys.GetEnumerator())
                using (var iv = values.GetEnumerator())
                    while (ik.MoveNext() && iv.MoveNext())
                    {
                        if (values != null)
                            mapProto.KeyValuePairs.Add(MakePair(ik.Current, iv.Current));
                        else
                            mapProto.KeyValuePairs.Add(MakePair(ik.Current, default(string)));
                    }
            }

            var valueProto = new LotusvNext.Expressions.ValueProto();
            valueProto.Map = mapProto;

            var expProto = new LotusvNext.Expressions.Expression();
            expProto.Literal = valueProto;
            return expProto;
        }
        public static LotusvNext.Expressions.Expression MakeDictionary(IEnumerable<long> keys=null, IEnumerable<long> values=null)
        {
            var mapProto = new LotusvNext.Expressions.ValueProto.Types.MapProto();
            if (keys != null && values != null)
            {
                using (var ik = keys.GetEnumerator())
                using (var iv = values.GetEnumerator())
                    while (ik.MoveNext() && iv.MoveNext())
                    {
                        if (values != null)
                            mapProto.KeyValuePairs.Add(MakePair(ik.Current, iv.Current));
                        else
                            mapProto.KeyValuePairs.Add(MakePair(ik.Current, default(long)));
                    }
            }

            var valueProto = new LotusvNext.Expressions.ValueProto();
            valueProto.Map = mapProto;

            var expProto = new LotusvNext.Expressions.Expression();
            expProto.Literal = valueProto;
            return expProto;
        }
        public static LotusvNext.Expressions.Expression MakeDictionary(IEnumerable<long> keys=null, IEnumerable<float> values=null)
        {
            var mapProto = new LotusvNext.Expressions.ValueProto.Types.MapProto();
            if (keys != null && values != null)
            {
                using (var ik = keys.GetEnumerator())
                using (var iv = values.GetEnumerator())
                    while (ik.MoveNext() && iv.MoveNext())
                    {
                        if (values != null)
                            mapProto.KeyValuePairs.Add(MakePair(ik.Current, iv.Current));
                        else
                            mapProto.KeyValuePairs.Add(MakePair(ik.Current, default(float)));
                    }
            }

            var valueProto = new LotusvNext.Expressions.ValueProto();
            valueProto.Map = mapProto;

            var expProto = new LotusvNext.Expressions.Expression();
            expProto.Literal = valueProto;
            return expProto;
        }
        public static LotusvNext.Expressions.Expression MakeFunRef(string name)
        {
            return new LotusvNext.Expressions.Expression()
            {
                Funcref = new LotusvNext.Expressions.FunctionReference()
                {
                    Name = name
                }
            };
        }
        public static LotusvNext.Expressions.Expression MakeVarRef(string name)
        {
            return new LotusvNext.Expressions.Expression()
            {
                Variable = new LotusvNext.Expressions.VariableReference()
                {
                    Name = name
                }
            };
        }
        public static LotusvNext.Expressions.Let.Types.Binding MakeBinding(string name, long initialValue)
        {
            return new LotusvNext.Expressions.Let.Types.Binding()
            {
                Name = name,
                InitialValue = Make(initialValue)
            };
        }
        public static LotusvNext.Expressions.Let.Types.Binding MakeBinding(string name, float initialValue)
        {
            return new LotusvNext.Expressions.Let.Types.Binding()
            {
                Name = name,
                InitialValue = Make(initialValue)
            };
        }
        public static LotusvNext.Expressions.Let.Types.Binding MakeBinding(string name, string initialValue)
        {
            return new LotusvNext.Expressions.Let.Types.Binding()
            {
                Name = name,
                InitialValue = Make(initialValue)
            };
        }
        public static LotusvNext.Expressions.Let.Types.Binding MakeBinding(string name, LotusvNext.Expressions.Expression initialValue)
        {
            return new LotusvNext.Expressions.Let.Types.Binding()
            {
                Name = name,
                InitialValue = initialValue
            };
        }
        public static LotusvNext.Expressions.Expression MakeLet(string name, LotusvNext.Expressions.Expression initialValue)
        {
            // Make Binding
            var bindingProto = MakeBinding(name, initialValue);

            // Make Let
            var letProto = new LotusvNext.Expressions.Let();
            letProto.VariableBindings.Add(bindingProto);

            // Make Expression
            return new LotusvNext.Expressions.Expression()
            {
                Let = letProto
            };
        }
        public static LotusvNext.Expressions.Expression MakeSet(string name, LotusvNext.Expressions.Expression value)
        {
            return new LotusvNext.Expressions.Expression()
            {
                Set = new LotusvNext.Expressions.Set()
                {
                    Name = name,
                    Value = value
                }
            };
        }
        // Helper functions for creating types
        public static LotusvNext.Types.TypeProto MakeInt64Type()
        {
            return new LotusvNext.Types.TypeProto()
            {
                ScalarType = new LotusvNext.Types.TypeProto.Types.Scalar()
                {
                    DataType = ONNX.TensorProto.Types.DataType.Int64
                }
            };
        }
        public static LotusvNext.Types.TypeProto MakeFloatType()
        {
            return new LotusvNext.Types.TypeProto()
            {
                ScalarType = new LotusvNext.Types.TypeProto.Types.Scalar()
                {
                    DataType = ONNX.TensorProto.Types.DataType.Float
                }
            };
        }
        public static LotusvNext.Types.TypeProto MakeStringType()
        {
            return new LotusvNext.Types.TypeProto()
            {
                ScalarType = new LotusvNext.Types.TypeProto.Types.Scalar()
                {
                    DataType = ONNX.TensorProto.Types.DataType.String
                }
            };
        }
        public static ONNX.TensorShapeProto MakeTensorShape(IEnumerable<long> dims)
        {
            var shapeProto = new ONNX.TensorShapeProto();

            if (dims == null)
                return shapeProto;

            // Scan through the input dimension list and copy the stored information into ONNX's shape object
            foreach (var d in dims)
            {
                var dimProto = new ONNX.TensorShapeProto.Types.Dimension();
                // For a dimension value, zero or negative dimension means variable-length dimension (aka unknown size)
                // while positive integer provides size explicitly.
                if (d < 1)
                    dimProto.DimParam = "None";
                else
                    dimProto.DimValue = d;
                shapeProto.Dim.Add(dimProto);
            }

            return shapeProto;
        }
        public static LotusvNext.Types.TypeProto MakeTensorType(
            ONNX.TensorProto.Types.DataType type, IEnumerable<long> dims)
        {
            return new LotusvNext.Types.TypeProto
            {
                TensorType = new LotusvNext.Types.TypeProto.Types.Tensor()
                {
                    ElemType = type,
                    Shape = MakeTensorShape(dims)
                }
            };
        }
        public static LotusvNext.Types.TypeProto MakeInt64TensorType(long dim)
        {
            return MakeTensorType(ONNX.TensorProto.Types.DataType.Int64, new List<long>() { dim });
        }
        public static LotusvNext.Types.TypeProto MakeFloatTensorType(long dim)
        {
            return MakeTensorType(ONNX.TensorProto.Types.DataType.Float, new List<long>() { dim });
        }
        public static LotusvNext.Types.TypeProto MakeStringTensorType(long dim)
        {
            return MakeTensorType(ONNX.TensorProto.Types.DataType.String, new List<long>() { dim });
        }
        public static LotusvNext.Types.TypeProto MakeInt64TensorType(IEnumerable<long> dims)
        {
            return MakeTensorType(ONNX.TensorProto.Types.DataType.Int64, dims);
        }
        public static LotusvNext.Types.TypeProto MakeFloatTensorType(IEnumerable<long> dims)
        {
            return MakeTensorType(ONNX.TensorProto.Types.DataType.Float, dims);
        }
        public static LotusvNext.Types.TypeProto MakeStringTensorType(IEnumerable<long> dims)
        {
            return MakeTensorType(ONNX.TensorProto.Types.DataType.String, dims);
        }
        public static LotusvNext.Types.TypeProto MakeMapType(
            ONNX.TensorProto.Types.DataType keyType, LotusvNext.Types.TypeProto valType)
        {
            return new LotusvNext.Types.TypeProto()
            {
                MapType = new LotusvNext.Types.TypeProto.Types.Map()
                {
                    KeyType = keyType,
                    ValueType = valType
                }
            };
        }
        public static LotusvNext.Types.TypeProto MakeStringToInt64MapType()
        {
            var keyType = ONNX.TensorProto.Types.DataType.String;
            var valType = MakeInt64Type();
            return MakeMapType(keyType, valType);
        }
        public static LotusvNext.Types.TypeProto MakeInt64ToStringMapType()
        {
            var keyType = ONNX.TensorProto.Types.DataType.Int64;
            var valType = MakeStringType();
            return MakeMapType(keyType, valType);
        }
        public static LotusvNext.Types.TypeProto MakeInt64ToInt64gMapType()
        {
            var keyType = ONNX.TensorProto.Types.DataType.Int64;
            var valType = MakeInt64Type();
            return MakeMapType(keyType, valType);
        }
        public static LotusvNext.Types.TypeProto MakeStringToStringMapType()
        {
            var keyType = ONNX.TensorProto.Types.DataType.String;
            var valType = MakeStringType();
            return MakeMapType(keyType, valType);
        }
        public static LotusvNext.Types.TypeProto MakeStringToFloatMapType()
        {
            var keyType = ONNX.TensorProto.Types.DataType.String;
            var valType = MakeFloatType();
            return MakeMapType(keyType, valType);
        }
        public static LotusvNext.Types.TypeProto MakeInt64ToFloatMapType()
        {
            var keyType = ONNX.TensorProto.Types.DataType.Int64;
            var valType = MakeFloatType();
            return MakeMapType(keyType, valType);
        }
        public static LotusvNext.Types.TypeProto.Types.ParameterDeclProto MakeParam(
            string name, LotusvNext.Types.TypeProto type)
        {
            return new LotusvNext.Types.TypeProto.Types.ParameterDeclProto()
            {
                Name = name,
                Type = type
            };
        }
        public static LotusvNext.Types.TypeProto.Types.ParameterDeclProto MakeInt64Param(string name)
        {
            return MakeParam(name, MakeInt64Type());
        }
        public static LotusvNext.Types.TypeProto.Types.ParameterDeclProto MakeFloatParam(string name)
        {
            return MakeParam(name, MakeFloatType());
        }
        public static LotusvNext.Types.TypeProto.Types.ParameterDeclProto MakeStringParam(string name)
        {
            return MakeParam(name, MakeStringType());
        }
        public static LotusvNext.Types.TypeProto.Types.ParameterDeclProto MakeInt64sParam(string name, long dim)
        {
            return MakeParam(name, MakeInt64TensorType(new List<long>() { dim }));
        }
        public static LotusvNext.Types.TypeProto.Types.ParameterDeclProto MakeFloatsParam(string name, long dim)
        {
            return MakeParam(name, MakeFloatTensorType(new List<long>() { dim }));
        }
        public static LotusvNext.Types.TypeProto.Types.ParameterDeclProto MakeStringsParam(string name, long dim)
        {
            return MakeParam(name, MakeStringTensorType(new List<long>() { dim }));
        }
        public static LotusvNext.Types.TypeProto MakeType(ColumnType type)
        {
            var typeProto = new LotusvNext.Types.TypeProto();

            DataKind rawKind;
            if (type.IsVector)
                rawKind = type.AsVector.ItemType.RawKind;
            else if (type.IsKey)
                rawKind = type.AsKey.RawKind;
            else
                rawKind = type.RawKind;

            if (type.IsVector)
            {
                var dims = new List<long>();
                if (type.IsVector)
                {
                    if (type.ValueCount == 0) // Unknown size.
                        dims.Add(-1);
                    else if (type.ValueCount == 1)
                        dims.Add(1);
                    else if (type.ValueCount > 1)
                    {
                        var vec = type.AsVector;
                        for (int i = 0; i < vec.DimCount; i++)
                            dims.Add(vec.GetDim(i));
                    }
                }
                switch (rawKind)
                {
                    case DataKind.BL:
                        typeProto = MakeInt64TensorType(dims);
                        break;
                    case DataKind.TX:
                        typeProto = MakeStringTensorType(dims);
                        break;
                    case DataKind.U4:
                        typeProto = MakeInt64TensorType(dims);
                        break;
                    case DataKind.R4:
                        typeProto = MakeFloatTensorType(dims);
                        break;
                    default:
                        Contracts.Assert(false, "Unknown type.");
                        break;
                }
            }
            else
            {
                switch (rawKind)
                {
                    case DataKind.BL:
                        typeProto = MakeInt64Type();
                        break;
                    case DataKind.TX:
                        typeProto = MakeStringType();
                        break;
                    case DataKind.U4:
                        typeProto = MakeInt64Type();
                        break;
                    case DataKind.R4:
                        typeProto = MakeFloatType();
                        break;
                    default:
                        Contracts.Assert(false, "Unknown type.");
                        break;
                }
            }
            return typeProto;
        }
        public static LotusvNext.Types.TypeProto.Types.SignatureDeclProto MakeSignature(
            IEnumerable<LotusvNext.Types.TypeProto.Types.ParameterDeclProto> inputs,
            IEnumerable<LotusvNext.Types.TypeProto.Types.ParameterDeclProto> outputs)
        {
            var signatureProto = new LotusvNext.Types.TypeProto.Types.SignatureDeclProto();
            signatureProto.InputParams.AddRange(inputs);
            signatureProto.OutputParams.AddRange(outputs);
            return signatureProto;
        }
        public static LotusvNext.Expressions.Expression MakeElementAccess(
            LotusvNext.Expressions.Expression container,
            IEnumerable<LotusvNext.Expressions.Expression> path,
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
        // Function-related helpers
        public static LotusvNext.FunctionDefProto MakeFunction(
            string name,
            LotusvNext.Types.TypeProto.Types.ParameterDeclProto input,
            LotusvNext.Types.TypeProto.Types.ParameterDeclProto output,
            IEnumerable<LotusvNext.Expressions.Expression> body=null)
        {
            var functionDefProto = new LotusvNext.FunctionDefProto();
            functionDefProto.InputParams.Add(input);
            functionDefProto.OutputParams.Add(output);
            if (body != null)
                functionDefProto.Body.AddRange(body);
            return functionDefProto;
        }
        public static LotusvNext.Expressions.Expression MakeLambda(
            LotusvNext.Types.TypeProto.Types.ParameterDeclProto input,
            LotusvNext.Types.TypeProto.Types.ParameterDeclProto output,
            IEnumerable<LotusvNext.Expressions.Expression> body=null)
        {
            var lambdaProto = new LotusvNext.Expressions.Lambda();
            lambdaProto.InputParams.Add(input);
            lambdaProto.OutputParams.Add(output);
            if (body != null)
                lambdaProto.Body.AddRange(body);
            return new LotusvNext.Expressions.Expression()
            {
                Lambda = lambdaProto
            };
        }
        // Model-related helper
        public static LotusvNext.OperatorSetIdProto MakeOperatorSetId(string domain, long version)
        {
            return new LotusvNext.OperatorSetIdProto()
            {
                Domain = domain,
                Version = version
            };
        }
        public static LotusvNext.FunctionalModelProto MakeFunctionalModel(
            string name,
            LotusvNext.Types.TypeProto.Types.SignatureDeclProto signature,
            IEnumerable<LotusvNext.Expressions.Expression> body,
            Dictionary<string, LotusvNext.FunctionDefProto> functions,
            Dictionary<string, LotusvNext.Types.TypeProto> types)
        {
            var functionalModelProto = new LotusvNext.FunctionalModelProto();
            functionalModelProto.Signature = signature;
            functionalModelProto.Name = name;
            functionalModelProto.Types_.Add(types);
            functionalModelProto.Functions.Add(functions);
            functionalModelProto.Body.AddRange(body);
            return functionalModelProto;
        }
        public static LotusvNext.ModelProto MakeModel(LotusvNext.FunctionalModelProto model)
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
        // Call-related helpers
        public static LotusvNext.Expressions.Call MakeCallProto(
            LotusvNext.Expressions.Expression target,
            IEnumerable<LotusvNext.Expressions.Expression> positional)
        {
            var callProto = new LotusvNext.Expressions.Call();
            callProto.Target = target; // Assume that it's always a function reference
            callProto.Positional.AddRange(positional);
            return callProto;
        }
        public static LotusvNext.Expressions.Expression Call(
            string name, IEnumerable<LotusvNext.Expressions.Expression> positional)
        {
            var funRef = MakeFunRef(name);
            return new LotusvNext.Expressions.Expression()
            {
                Call = MakeCallProto(funRef, positional)
            };
        }
        public static LotusvNext.Expressions.Expression Call(
            string name, params LotusvNext.Expressions.Expression[] positional)
        {
            var funRef = MakeFunRef(name);
            return new LotusvNext.Expressions.Expression()
            {
                Call = MakeCallProto(funRef, positional)
            };
        }
        public static LotusvNext.Expressions.Expression MakeComment(string text)
        {
            var comment = new LotusvNext.Expressions.Comment();
            comment.Comment_.Add(text);
            return new LotusvNext.Expressions.Expression()
            {
                Comment = comment
            };
        }
        // For-related helper
        public static LotusvNext.Expressions.For MakeForProto(
            LotusvNext.Expressions.Let.Types.Binding induction,
            LotusvNext.Expressions.Expression condition,
            LotusvNext.Expressions.Set step,
            IEnumerable<LotusvNext.Expressions.Expression> body=null)
        {
            var forProto = new LotusvNext.Expressions.For();
            forProto.InductionVariables.Add(induction);
            forProto.Condition = condition;
            if (body != null)
                forProto.Body.AddRange(body);
            forProto.Step.Add(step);
            return forProto;
        }
        // If-related helper
        public static LotusvNext.Expressions.Cond.Types.IfThen MakeIfThen(
            LotusvNext.Expressions.Expression cond, IEnumerable<LotusvNext.Expressions.Expression> todo=null)
        {
            var ifThenProto = new LotusvNext.Expressions.Cond.Types.IfThen();
            ifThenProto.Condition = cond;
            if (todo != null)
                ifThenProto.ThenClause.AddRange(todo);
            return ifThenProto;
        }
        public static LotusvNext.Expressions.Expression MakeIf(
            LotusvNext.Expressions.Expression cond,
            IEnumerable<LotusvNext.Expressions.Expression> todo=null)
        {
            return new LotusvNext.Expressions.Expression()
            {
                IfThen = new LotusvNext.Expressions.If()
                {
                    IfThen = MakeIfThen(cond, todo)
                }
            };
        }
        public static LotusvNext.Expressions.Expression MakeIf(
            LotusvNext.Expressions.Expression cond,
            params LotusvNext.Expressions.Expression[] todo)
        {
            return new LotusvNext.Expressions.Expression()
            {
                IfThen = new LotusvNext.Expressions.If()
                {
                    IfThen = MakeIfThen(cond, todo)
                }
            };
        }
        // Misc
        public static IEnumerable<LotusvNext.Expressions.Expression> Range(long start, long end, LotusvNext.Expressions.Expression biasExp=null)
        {
            var rangeExps = new List<LotusvNext.Expressions.Expression>();
            for (long i=start; i < end; ++i)
            {
                if (biasExp == null)
                    rangeExps.Add(Make(i));
                else
                    rangeExps.Add(Call("Add", biasExp, Make(i)));

            }
            return rangeExps;
        }
    }
}