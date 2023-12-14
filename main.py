import pathlib
from Tests.test_Ident_method import test_ControllableReLiNetIdentModel
from Tests.test_SVD_method import test_ControllableReLiNetSVDModel
from Tests.test_2MM_method import test_ControllableReLiNet2MMModel

def main() -> None:
    tmp_path = pathlib.Path('tmp')
    tmp_path.mkdir(exist_ok=True)
    test_ControllableReLiNetIdentModel(tmp_path)
    "test_ControllableReLiNetSVDModel(tmp_path)"
    test_ControllableReLiNet2MMModel(tmp_path)


if __name__ == '__main__':
    main()
