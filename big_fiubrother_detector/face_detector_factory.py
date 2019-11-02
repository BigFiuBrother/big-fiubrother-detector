import os
import yaml


class FaceDetectorFactory:

    @staticmethod
    def build(configuration):

        face_detector_type = configuration['type']

        if face_detector_type == "caffe_mtcnn":
            from big_fiubrother_detector.face_detector_caffe_mtcnn import FaceDetectorCaffeMTCNN
            return FaceDetectorCaffeMTCNN()

        elif face_detector_type == "movidius_mtcnn":
            from big_fiubrother_detector.face_detector_movidius_mtcnn import FaceDetectorMovidiusMTCNN
            return FaceDetectorMovidiusMTCNN(configuration['movidius_id_pnet'], configuration['movidius_id_onet'])

        elif face_detector_type == "movidius_ssd":
            from big_fiubrother_detector.face_detector_movidius_ssd import FaceDetectorMovidiusSSD
            return FaceDetectorMovidiusSSD(configuration['movidius_id'], configuration['longrange'])

    @staticmethod
    def build_movidius_ssd():
        config_path = os.path.dirname(os.path.realpath(__file__)) + "/../config/config_mvds_ssd.yaml"
        return FaceDetectorFactory.build(FaceDetectorFactory._read_config_file(config_path))

    @staticmethod
    def build_movidius_ssd_longrange():
        config_path = os.path.dirname(os.path.realpath(__file__)) + "/../config/config_mvds_ssd_longrange.yaml"
        return FaceDetectorFactory.build(FaceDetectorFactory._read_config_file(config_path))

    @staticmethod
    def build_movidius_mtcnn():
        config_path = os.path.dirname(os.path.realpath(__file__)) + "/../config/config_mvds_mtcnn.yaml"
        return FaceDetectorFactory.build(FaceDetectorFactory._read_config_file(config_path))

    @staticmethod
    def build_caffe_mtcnn():
        config_path = os.path.dirname(os.path.realpath(__file__)) + "/../config/config_caffe_mtcnn.yaml"
        return FaceDetectorFactory.build(FaceDetectorFactory._read_config_file(config_path))

    @staticmethod
    def _read_config_file(path):
        with open(path) as config_file:
            configuration = yaml.load(config_file)
            return configuration['face_detector']
