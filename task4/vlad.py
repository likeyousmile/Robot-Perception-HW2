import pickle
import numpy as np
import cv2 as cv
from tqdm.contrib.concurrent import process_map
from multiprocessing import cpu_count
import time

DESCRIPTOR = {
    "ORB": (cv.ORB_create, 32),
    "SIFT": (cv.SIFT_create, 128)
}


def time_cost(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"time cost: {int(time.time() - start)}s")
        return result

    return wrapper


@time_cost
def multiprocess(func, *iterable):
    chunk_size, extra = divmod(len(iterable[0]), cpu_count() * 4)
    if extra:
        chunk_size += 1
    return process_map(func, *iterable, chunksize=chunk_size)


class VLAD(object):
    def __init__(self, labels=None, centers=None, vlad=None, file_list=None, codebook_dim=None, descriptor_type=None):
        self.labels = labels
        self.centers = centers
        self.vlad = vlad
        self.file_list = file_list
        self.codebook_dim = codebook_dim
        self.descriptor_type = descriptor_type
        self.detector, self.descriptor_dim = DESCRIPTOR[descriptor_type]

    def extract_descriptor(self, image):
        detector = self.detector()
        _key_point, descriptor = detector.detectAndCompute(image, None)
        return descriptor

    @time_cost
    def extract_codebook(self, descriptors):
        _ret_val, labels, centers = cv.kmeans(
            data=np.concatenate(descriptors).astype(np.float32),
            K=self.codebook_dim,
            bestLabels=None,
            criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 0.01),
            attempts=3,
            flags=cv.KMEANS_RANDOM_CENTERS
        )
        cursor = 0
        label_list = []
        for desc in descriptors:
            label_list.append(labels[cursor: cursor + len(desc)])
            cursor += len(desc)
        self.labels = label_list
        self.centers = centers

    def extract_vlad(self, descriptor, center_id):
        center = self.centers[center_id]

        vlad_base = np.zeros(shape=[self.codebook_dim, self.descriptor_dim])
        for idx in range(len(descriptor)):
            vlad_base[center_id[idx]] = vlad_base[center_id[idx]] + descriptor[idx] - center[idx]

        vlad_norm = vlad_base.copy()
        cv.normalize(vlad_base, vlad_norm, 1.0, 0.0, cv.NORM_L2)

        return vlad_norm.reshape(self.codebook_dim * self.descriptor_dim, -1)

    def calculate_vlad(self, descriptor):
        vlad_base = np.zeros(shape=[self.codebook_dim, self.descriptor_dim])

        for vec in descriptor:
            dist = np.linalg.norm(self.centers - vec, axis=1)
            min_idx = dist.argmin()
            vlad_base[min_idx] = vlad_base[min_idx] + vec - self.centers[min_idx]

        vlad_norm = vlad_base.copy()
        cv.normalize(vlad_base, vlad_norm, 1.0, 0.0, cv.NORM_L2)
        vlad_norm = vlad_norm.reshape(self.codebook_dim * self.descriptor_dim, -1)
        return vlad_norm

    @classmethod
    def create(cls, database_dir: str, codebook_dim: int, descriptor_type: str):
        file_list = [f"{database_dir}/image{n}.png" for n in range(28600)]
        model = cls(file_list=file_list, codebook_dim=codebook_dim, descriptor_type=descriptor_type)
        print("reading images")
        images = multiprocess(cv.imread, model.file_list)

        print("extracting descriptors")
        descriptors = multiprocess(model.extract_descriptor, images)

        print("extracting codebook")
        model.extract_codebook(descriptors)

        print("extracting VLAD matrix")
        model.vlad = multiprocess(model.extract_vlad, descriptors, model.labels)
        return model

    def save(self, model_path: str):
        with open(model_path, "wb") as f:
            attrs = {
                "labels": self.labels,
                "centers": self.centers,
                "vlad": self.vlad,
                "file_list": self.file_list,
                "codebook_dim": self.codebook_dim,
                "descriptor_type": self.descriptor_type
            }
            pickle.dump(attrs, f)

    @classmethod
    def load(cls, model_path: str):
        with open(model_path, "rb") as f:
            attrs = pickle.load(f)
        return cls(**attrs)

    def query(self, file_path: str, k: int):
        image = cv.imread(file_path)
        descriptor = self.extract_descriptor(image)
        query_vlad = self.calculate_vlad(descriptor)

        distance = np.linalg.norm(self.vlad - query_vlad, axis=1).squeeze()
        similar_idx = distance.argsort()[:k]
        similar_files = [self.file_list[idx] for idx in similar_idx]
        return similar_files


if __name__ == "__main__":
    vlad_orb = VLAD.create("F:\\GradSemester1\\Y1\\Robot_perception\\HW2\\HW2-TheDevilIsinTheDetail\\database", codebook_dim=32, descriptor_type="ORB")
    vlad_orb.save("./VLAD-ORB-32.pkl")
