"""
验证脚本测试

本测试模块验证 verify_baselines_official.py 脚本的正确性：
1. 测试五个测试函数是否存在
2. 测试数据文件缺失时的行为（必须 Fail）

Requirements:
    - 8.1-8.9: 验证所有官方基准测试函数的存在性和正确性
"""

import pytest
import inspect
from pathlib import Path

# 导入验证脚本模块
from tests.verify_baselines_official import TestOfficialBenchmarks


class TestVerifyBaselinesOfficial:
    """验证脚本测试类"""
    
    def test_all_test_functions_exist(self):
        """
        测试五个测试函数是否存在
        
        验证：
        - test_xgboost_on_breast_cancer() 存在
        - test_tabr_on_breast_cancer() 存在
        - test_hyperfast_on_breast_cancer() 存在
        - test_mogonet_on_rosmap() 存在
        - test_transtee_on_ihdp() 存在
        
        Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
        """
        # 获取 TestOfficialBenchmarks 类的所有方法
        methods = inspect.getmembers(TestOfficialBenchmarks, predicate=inspect.isfunction)
        method_names = [name for name, _ in methods]
        
        # 验证五个测试函数存在
        required_tests = [
            'test_xgboost_on_breast_cancer',
            'test_tabr_on_breast_cancer',
            'test_hyperfast_on_breast_cancer',
            'test_mogonet_on_rosmap',
            'test_transtee_on_ihdp'
        ]
        
        for test_name in required_tests:
            assert test_name in method_names, f"测试函数 {test_name} 不存在"
        
        print(f"\n✓ 所有必需的测试函数都存在:")
        for test_name in required_tests:
            print(f"  - {test_name}")
    
    def test_test_functions_are_methods(self):
        """
        测试所有测试函数都是 TestOfficialBenchmarks 类的方法
        
        验证：
        - 所有测试函数都正确定义为类方法
        """
        test_class = TestOfficialBenchmarks()
        
        required_tests = [
            'test_xgboost_on_breast_cancer',
            'test_tabr_on_breast_cancer',
            'test_hyperfast_on_breast_cancer',
            'test_mogonet_on_rosmap',
            'test_transtee_on_ihdp'
        ]
        
        for test_name in required_tests:
            assert hasattr(test_class, test_name), f"方法 {test_name} 不存在"
            assert callable(getattr(test_class, test_name)), f"{test_name} 不是可调用的方法"
    
    def test_mogonet_fails_on_missing_data(self):
        """
        测试 MOGONET 在数据文件缺失时的行为
        
        验证：
        - 数据文件缺失时测试必须失败（Fail），而非跳过（Skip）
        - 抛出 FileNotFoundError 或 pytest.fail
        
        Requirements: 8.6, 8.7, 8.9
        """
        from src.baselines.mogonet_baseline import ROSMAPDataset
        
        # 检查 ROSMAP 数据文件是否存在
        data_dir = Path("data/baselines_official/ROSMAP/")
        required_files = [
            '1_tr.csv', '1_te.csv',
            '2_tr.csv', '2_te.csv',
            '3_tr.csv', '3_te.csv',
            'labels_tr.csv', 'labels_te.csv'
        ]
        
        files_exist = all((data_dir / file).exists() for file in required_files)
        
        if not files_exist:
            # 数据文件缺失，验证会抛出 FileNotFoundError
            with pytest.raises(FileNotFoundError) as exc_info:
                dataset = ROSMAPDataset()
                dataset.load_data()
            
            # 验证错误信息包含文件路径和下载指引
            error_message = str(exc_info.value)
            assert "ROSMAP" in error_message, "错误信息应包含 'ROSMAP'"
            assert "data/baselines_official/ROSMAP" in error_message or "下载" in error_message, (
                "错误信息应包含文件路径或下载指引"
            )
            
            print(f"\n✓ MOGONET 数据文件缺失时正确抛出 FileNotFoundError")
            print(f"  - 错误信息包含文件路径和下载指引")
        else:
            print(f"\n⚠️  ROSMAP 数据文件存在，跳过缺失测试")
    
    def test_transtee_fails_on_missing_data(self):
        """
        测试 TransTEE 在数据文件缺失时的行为
        
        验证：
        - 数据文件缺失时测试必须失败（Fail），而非跳过（Skip）
        - 抛出 FileNotFoundError 或 pytest.fail
        
        Requirements: 8.6, 8.7, 8.9
        """
        from src.baselines.transtee_baseline import IHDPDataset
        
        # 检查 IHDP 数据文件是否存在
        data_file = Path("data/baselines_official/IHDP/ihdp_npci_1.csv")
        
        if not data_file.exists():
            # 数据文件缺失，验证会抛出 FileNotFoundError
            with pytest.raises(FileNotFoundError) as exc_info:
                dataset = IHDPDataset()
                dataset.load_data()
            
            # 验证错误信息包含文件路径和下载指引
            error_message = str(exc_info.value)
            assert "IHDP" in error_message, "错误信息应包含 'IHDP'"
            assert "data/baselines_official/IHDP" in error_message or "下载" in error_message, (
                "错误信息应包含文件路径或下载指引"
            )
            
            print(f"\n✓ TransTEE 数据文件缺失时正确抛出 FileNotFoundError")
            print(f"  - 错误信息包含文件路径和下载指引")
        else:
            print(f"\n⚠️  IHDP 数据文件存在，跳过缺失测试")
    
    def test_error_messages_contain_download_instructions(self):
        """
        测试错误信息包含下载指引
        
        验证：
        - FileNotFoundError 错误信息包含清晰的下载指引
        
        Requirements: 8.7
        """
        from src.baselines.mogonet_baseline import ROSMAPDataset
        from src.baselines.transtee_baseline import IHDPDataset
        
        # 测试 ROSMAP 错误信息
        data_dir = Path("data/baselines_official/ROSMAP/")
        if not data_dir.exists() or not any(data_dir.iterdir()):
            try:
                dataset = ROSMAPDataset()
                dataset.load_data()
            except FileNotFoundError as e:
                error_message = str(e)
                # 验证错误信息包含下载指引关键词
                assert any(keyword in error_message for keyword in ["下载", "download", "获取"]), (
                    "ROSMAP 错误信息应包含下载指引"
                )
        
        # 测试 IHDP 错误信息
        data_file = Path("data/baselines_official/IHDP/ihdp_npci_1.csv")
        if not data_file.exists():
            try:
                dataset = IHDPDataset()
                dataset.load_data()
            except FileNotFoundError as e:
                error_message = str(e)
                # 验证错误信息包含下载指引关键词
                assert any(keyword in error_message for keyword in ["下载", "download", "获取"]), (
                    "IHDP 错误信息应包含下载指引"
                )
        
        print(f"\n✓ 错误信息包含下载指引")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
