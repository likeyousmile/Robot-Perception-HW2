from vlad import VLAD
import numpy as np
import cv2 as cv


class Visualization(object):
    def __init__(self, model):
        self.model: VLAD = model
        self.files = []

    def callback(self, mouse_event, x, y, _flags, _param):
        if mouse_event != 4:
            return
        choose_idx = x // 100 + y // 100 * 5
        self.files = self.model.query(self.files[choose_idx], 25)
        self.show()

    def search(self, query_file: str):
        self.files = self.model.query(query_file, 25)
        cv.namedWindow("view")
        cv.setMouseCallback("view", self.callback)
        self.show()
        cv.waitKey(0)
        cv.destroyAllWindows()
        cv.waitKey(1)
        print(self.files[0])

    def show(self):
        images = [cv.imread(filename) for filename in self.files]
        image = self.make_grid(
            images,
            cols=int(np.sqrt(len(images)))
        )
        cv.imshow("view", image)

    @staticmethod
    def make_grid(images, cols, padding=2):
        n_images = len(images)
        rows, extra = divmod(n_images, cols)
        if extra:
            rows += 1

        height, width = images[0].shape[:2]
        shape = np.array(images[0].shape) * \
                np.array((rows, cols, 1)) + \
                np.array((padding * (rows - 1), padding * (cols - 1), 0))
        canvas = np.full(shape, padding, dtype=images[0].dtype)

        k = 0
        for y in range(rows):
            for x in range(cols):
                if k >= n_images:
                    break
                canvas[y * (height + padding): (y + 1) * (height + padding) - padding,
                x * (width + padding): (x + 1) * (width + padding) - padding] = images[k]
                k = k + 1

        return canvas


if __name__ == "__main__":
    vlad_orb = VLAD.load("./VLAD-ORB-32.pkl")

    view = Visualization(model=vlad_orb)
    view.search("F:\\GradSemester1\\Y1\\Robot_perception\\HW2\\HW2-TheDevilIsinTheDetail\\queries\\query4.png")

    # query1 - image2159:  E
    # query2 - image3668:  U
    # query3 - image6837:  R
    # query4 - image27374: A
    # query5 - image14461: S

