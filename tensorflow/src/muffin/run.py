from pathlib import Path
from typing import List, Optional
import datetime

from utils.db_manager import DbManager
from utils.utils import get_HH_mm_ss
from src.cases_generation.model_info_generator import ModelInfoGenerator
from src.cases_generation.model_generator import ModelGenerator
from src.cases_generation.data_generator import DataGenerator
from utils.selection import Roulette


class TrainingDebugger(object):

    def __init__(self, config: dict, use_heuristic: bool = True, generate_mode: str = 'template', timeout: float = 60):
        super().__init__()
        self.__output_dir = config['output_dir']
        self.__db_manager = DbManager(config['db_path'])
        # Roulette处理器
        from utils.utils import layer_types, layer_conditions
        self.__selector = Roulette(layer_types=layer_types,
                                   layer_conditions=layer_conditions,
                                   use_heuristic=use_heuristic)
        self.__model_info_generator = ModelInfoGenerator(
            config['model'], self.__db_manager, self.__selector, generate_mode)
        self.__model_generator = ModelGenerator(
            config['model']['var']['weight_value_range'],
            config['backends'],
            self.__db_manager, self.__selector, timeout)
        self.__training_data_generator = DataGenerator(config['training_data'])

    def run_generation(
            self, model_info: Optional[dict] = None, initial_weight_dir: Optional[str] = None,
            dataset_name: Optional[str] = None):
        '''随机生成模型和数据, 可指定现成模型信息
        '''
        # 随机生成model
        print('model生成开始...')
        json_path, model_input_shapes, model_output_shapes, model_id, exp_dir = self.__model_info_generator.generate(
            save_dir=self.__output_dir, model_info=model_info)
        ok_backends = self.__model_generator.generate(json_path=json_path,
                                                      model_id=model_id,
                                                      exp_dir=exp_dir,
                                                      initial_weight_dir=initial_weight_dir)
        print(f'model生成完毕: model_id={model_id} ok_backends={ok_backends}')

        if len(ok_backends) >= 2:  # 否则没有继续实验的必要
            # 随机生成training data
            print('training data生成开始...')
            if dataset_name is None:
                self.__training_data_generator.generate(input_shapes=model_input_shapes,
                                                        output_shapes=model_output_shapes,
                                                        exp_dir=exp_dir)
            else:
                import shutil
                shutil.copytree(str(Path('dataset') / dataset_name), str(Path(exp_dir) / 'dataset'))
            print('training data生成完毕.')

        return model_id, exp_dir, ok_backends

    def run_generation_for_dataset(self, dataset_name: str):
        '''随机生成模型, 数据使用现成dataset
        '''
        # 随机生成model
        print('model生成开始...')
        json_path, _, _, model_id, exp_dir = self.__model_info_generator.generate_for_dataset(
            save_dir=self.__output_dir, dataset_name=dataset_name)
        ok_backends = self.__model_generator.generate(json_path=json_path,
                                                      model_id=model_id,
                                                      exp_dir=exp_dir)
        print(f'model生成完毕: model_id={model_id} ok_backends={ok_backends}')

        return model_id, exp_dir, ok_backends


def main(testing_config):
    config = {
        'model': {
            'var': {
                'tensor_dimension_range': (2, 5),
                'tensor_element_size_range': (2, 5),
                'weight_value_range': (-1.0, 1.0),
                'small_value_range': (0, 1),
                'vocabulary_size': 1001,
            },
            'node_num_range': (5, 5),
            'dag_io_num_range': (1, 3),
            'dag_max_branch_num': 2,
            'cell_num': 3,
            'node_num_per_normal_cell': 10,
            'node_num_per_reduction_cell': 2,
        },
        'training_data': {
            'instance_num': 10,
            'element_val_range': (0, 100),
        },
        'db_path': str(Path.cwd() / testing_config['data_dir'] / f'{testing_config["dataset_name"]}.db'),
        'output_dir': str(Path.cwd() / testing_config['data_dir'] / f'{testing_config["dataset_name"]}_output'),
        'report_dir': str(Path.cwd() / testing_config['data_dir'] / f'{testing_config["dataset_name"]}_report'),
        'backends': ['tensorflow'],
        'distance_threshold': 0,
    }

    DEBUG_MODE = testing_config["debug_mode"]
    CASE_NUM = testing_config["case_num"]
    TIMEOUT = testing_config["timeout"]  # 秒
    USE_HEURISTIC = bool(testing_config["use_heuristic"])  # 是否开启启发式规则
    GENERATE_MODE = testing_config["generate_mode"]  # seq\merging\dag\template

    debugger = TrainingDebugger(config, USE_HEURISTIC, GENERATE_MODE, TIMEOUT)
    start_time = datetime.datetime.now()

    for i in range(CASE_NUM):
        print(f"######## Round {i} ########")
        try:
            print("------------- generation -------------")
            model_id, exp_dir, ok_backends = debugger.run_generation_for_dataset(testing_config["dataset_name"])
        except Exception:
            import traceback
            traceback.print_exc()

    end_time = datetime.datetime.now()
    time_delta = end_time - start_time
    h, m, s = get_HH_mm_ss(time_delta)
    print(f"Time used: {h} hour,{m} min,{s} sec")


if __name__ == '__main__':
    import json
    with open(str("testing_config.json"), "r") as f:
        testing_config = json.load(f)
    main(testing_config)
