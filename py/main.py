import sys
import importlib
# add_test = importlib.import_module("..'cmake-build-debug'.add_test")

# sys.path.append("/Users/peng599/Library/CloudStorage/OneDrive-PNNL/Documents/pppp/CLion/ParallelSpGEMM/cmake-build-debug")
# print(sys.path)

# import add_test
# print(F"restuls: {add_test.add(1, 7)}")

# import add_test
# import ..build.add_test
# from ..build import add_test
# import build.add_test as add_test
# from ..build import add_test
# import ..build.add_test
# sys.path.append("../build")
sys.path.append("../cmake-build-debug")
# from build import add_test
# from build import add_test
# from build import add_test
import add_test
print(F"restuls: {add_test.add(1, 7)}")


