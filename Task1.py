import sys
import numpy as np
import tqdm
import open3d as o3d
import pickle
import plotly.graph_objects as go

np.errstate(divide="ignore", invalid='ignore')


class PointCloudSegmentation:

    def __init__(self, point_cloud):
        self.point_cloud = point_cloud
        self.planes = []
        self.threshold = 0

    def segment_plane(self, n_samples: int, threshold: float):
        self.threshold = threshold
        points = self.point_cloud.copy()
        epoch = 0
        while len(points) > 3:
            epoch += 1
            print(f"Epoch {epoch}: {len(points)} points left", flush=True)
            n, d = self._ransac(points, n_samples, threshold)
            self.planes.append((n, d))

            distance = np.abs(points @ n + d)
            points = points[distance > threshold]

        print(f"{len(self.planes)} planes found")

    def visualize(self):
        scatters = []
        for i in range(len(self.planes)):
            n, d = self.planes[i]
            distance = np.abs(self.point_cloud @ n + d)
            consensus_set = self.point_cloud[distance <= self.threshold]
            scatters.append(go.Scatter3d(
                x=consensus_set[:, 0],
                y=consensus_set[:, 1],
                z=consensus_set[:, 2],
                marker=go.scatter3d.Marker(size=1),
                opacity=0.6,
                mode="markers"
            ))
        fig = go.Figure(scatters)
        fig.show()

    def _ransac(self, point_cloud, max_iterations=10000, distance_threshold=0.01):
        score = 0
        plane = None

        for _ in tqdm.trange(max_iterations, file=sys.stdout):
            sample_points = point_cloud[np.random.choice(point_cloud.shape[0], 3, replace=False)]
            n, d = self.compute_triangle_plain(sample_points)

            distance = np.abs(point_cloud @ n + d)
            iter_score = np.sum(distance < distance_threshold)
            if iter_score > score:
                score = iter_score
                plane = n, d
        print(f"Plane: ({plane}), Score: {score}")
        return plane

    @staticmethod
    def compute_triangle_plain(points):
        n = np.cross(
            points[0] - points[2],
            points[1] - points[2]
        )

        n /= np.linalg.norm(n)
        d = - points[0] @ n

        return n, d

    def save(self, model_path: str):
        with open(model_path, "wb") as f:
            attrs = {
                "point_cloud": self.point_cloud,
                "planes": self.planes,
                "threshold": self.threshold
            }
            pickle.dump(attrs, f)

    @classmethod
    def load(cls, model_path: str):
        with open(model_path, "rb") as f:
            attrs = pickle.load(f)
        m = cls(attrs["point_cloud"])
        m.planes = attrs["planes"]
        m.point_sets = attrs["point_sets"]
        m.threshold = attrs["threshold"]
        return m


if __name__ == "__main__":
    ITERATIONS = 30000
    THRESHOLD = 0.03

    pcd = np.array(o3d.io.read_point_cloud("C:\\Users\\79260\\Desktop\\record_00348.pcd").points)

    model = PointCloudSegmentation(pcd)
    model.segment_plane(ITERATIONS, THRESHOLD)

    model.save(f"P3D-{ITERATIONS}-{THRESHOLD}.pkl")
    # model = PointCloudSegmentation.load(f"P3D-{ITERATIONS}-{THRESHOLD}.pkl")

    model.visualize()
