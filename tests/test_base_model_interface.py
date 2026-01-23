"""
BaseModel 接口测试模块

验证 BaseModel 抽象基类的接口完整性和正确性。
"""

import pytest
import inspect
from abc import ABC
from src.baselines.base_model import BaseModel


def test_base_model_is_abstract():
    """
    测试 BaseModel 是抽象类
    
    验证：
    - BaseModel 继承自 ABC
    - 不能直接实例化 BaseModel
    
    Validates: Requirements 1.1-1.6
    """
    # 验证 BaseModel 是抽象类
    assert issubclass(BaseModel, ABC), "BaseModel 应该继承自 ABC"
    
    # 验证不能直接实例化
    with pytest.raises(TypeError) as exc_info:
        BaseModel()
    
    assert "abstract" in str(exc_info.value).lower(), "应该抛出抽象类实例化错误"


def test_base_model_has_required_methods():
    """
    测试 BaseModel 包含所有必需的方法
    
    验证：
    - fit() 方法存在
    - predict() 方法存在
    - predict_proba() 方法存在
    - evaluate() 方法存在
    - get_params() 方法存在
    - set_params() 方法存在
    
    Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
    """
    required_methods = ['fit', 'predict', 'predict_proba', 'evaluate', 'get_params', 'set_params']
    
    for method_name in required_methods:
        assert hasattr(BaseModel, method_name), f"BaseModel 缺少方法: {method_name}"
        method = getattr(BaseModel, method_name)
        assert callable(method), f"{method_name} 应该是可调用的方法"


def test_base_model_abstract_methods():
    """
    测试 BaseModel 的抽象方法
    
    验证：
    - fit() 是抽象方法
    - predict() 是抽象方法
    - predict_proba() 是抽象方法
    - evaluate() 是抽象方法
    
    Validates: Requirements 1.1, 1.2, 1.3, 1.4
    """
    abstract_methods = ['fit', 'predict', 'predict_proba', 'evaluate']
    
    # 获取 BaseModel 的抽象方法
    base_abstract_methods = BaseModel.__abstractmethods__
    
    for method_name in abstract_methods:
        assert method_name in base_abstract_methods, f"{method_name} 应该是抽象方法"


def test_base_model_init():
    """
    测试 BaseModel 的初始化方法
    
    验证：
    - __init__ 方法接受 random_state 参数
    - 默认 random_state 为 42
    - model 属性初始化为 None
    
    Validates: Requirements 12.1
    """
    # 由于 BaseModel 是抽象类，我们创建一个简单的具体实现来测试
    class ConcreteModel(BaseModel):
        def fit(self, X, y):
            return self
        
        def predict(self, X):
            return None
        
        def predict_proba(self, X):
            return None
        
        def evaluate(self, X, y):
            return {}
    
    # 测试默认初始化
    model = ConcreteModel()
    assert model.random_state == 42, "默认 random_state 应该是 42"
    assert model.model is None, "model 属性应该初始化为 None"
    
    # 测试自定义 random_state
    model_custom = ConcreteModel(random_state=123)
    assert model_custom.random_state == 123, "应该能够设置自定义 random_state"


def test_base_model_get_params():
    """
    测试 BaseModel 的 get_params 方法
    
    验证：
    - get_params() 返回字典
    - 字典包含 'random_state' 键
    
    Validates: Requirements 1.5
    """
    class ConcreteModel(BaseModel):
        def fit(self, X, y):
            return self
        
        def predict(self, X):
            return None
        
        def predict_proba(self, X):
            return None
        
        def evaluate(self, X, y):
            return {}
    
    model = ConcreteModel(random_state=42)
    params = model.get_params()
    
    assert isinstance(params, dict), "get_params() 应该返回字典"
    assert 'random_state' in params, "参数字典应该包含 'random_state'"
    assert params['random_state'] == 42, "random_state 值应该正确"


def test_base_model_set_params():
    """
    测试 BaseModel 的 set_params 方法
    
    验证：
    - set_params() 能够设置参数
    - set_params() 返回 self（支持链式调用）
    
    Validates: Requirements 1.6
    """
    class ConcreteModel(BaseModel):
        def __init__(self, random_state=42, max_iter=100):
            super().__init__(random_state)
            self.max_iter = max_iter
        
        def fit(self, X, y):
            return self
        
        def predict(self, X):
            return None
        
        def predict_proba(self, X):
            return None
        
        def evaluate(self, X, y):
            return {}
    
    model = ConcreteModel()
    
    # 测试设置参数
    result = model.set_params(random_state=999, max_iter=500)
    
    assert result is model, "set_params() 应该返回 self"
    assert model.random_state == 999, "random_state 应该被更新"
    assert model.max_iter == 500, "max_iter 应该被更新"


def test_base_model_method_signatures():
    """
    测试 BaseModel 方法的签名
    
    验证：
    - fit(X, y) 接受两个参数
    - predict(X) 接受一个参数
    - predict_proba(X) 接受一个参数
    - evaluate(X, y) 接受两个参数
    
    Validates: Requirements 1.1, 1.2, 1.3, 1.4
    """
    # 检查 fit 方法签名
    fit_sig = inspect.signature(BaseModel.fit)
    fit_params = list(fit_sig.parameters.keys())
    assert 'X' in fit_params, "fit() 应该有 X 参数"
    assert 'y' in fit_params, "fit() 应该有 y 参数"
    
    # 检查 predict 方法签名
    predict_sig = inspect.signature(BaseModel.predict)
    predict_params = list(predict_sig.parameters.keys())
    assert 'X' in predict_params, "predict() 应该有 X 参数"
    
    # 检查 predict_proba 方法签名
    predict_proba_sig = inspect.signature(BaseModel.predict_proba)
    predict_proba_params = list(predict_proba_sig.parameters.keys())
    assert 'X' in predict_proba_params, "predict_proba() 应该有 X 参数"
    
    # 检查 evaluate 方法签名
    evaluate_sig = inspect.signature(BaseModel.evaluate)
    evaluate_params = list(evaluate_sig.parameters.keys())
    assert 'X' in evaluate_params, "evaluate() 应该有 X 参数"
    assert 'y' in evaluate_params, "evaluate() 应该有 y 参数"


def test_base_model_docstrings():
    """
    测试 BaseModel 的文档字符串
    
    验证：
    - 类有文档字符串
    - 所有方法都有文档字符串
    
    Validates: Requirements 10.3
    """
    # 检查类文档字符串
    assert BaseModel.__doc__ is not None, "BaseModel 应该有类文档字符串"
    assert len(BaseModel.__doc__.strip()) > 0, "类文档字符串不应为空"
    
    # 检查方法文档字符串
    methods = ['fit', 'predict', 'predict_proba', 'evaluate', 'get_params', 'set_params']
    
    for method_name in methods:
        method = getattr(BaseModel, method_name)
        assert method.__doc__ is not None, f"{method_name}() 应该有文档字符串"
        assert len(method.__doc__.strip()) > 0, f"{method_name}() 的文档字符串不应为空"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
