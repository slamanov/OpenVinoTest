from models_execution.basic_model_run import BasicModelRun


class AgeGenderRecognitionRetailModelRun(BasicModelRun):

    dim = (62, 62)

    def age_gender_detection(self, param, net_PVB, exec_net_PVB):
        pass