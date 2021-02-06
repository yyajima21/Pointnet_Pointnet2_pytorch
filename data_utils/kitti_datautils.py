import numpy as np
import os


class KittiDatasetPrep:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.room_idxs = sorted(os.listdir(self.root_dir))
        self.counter = 0
        self.resolution = 1
        self.rename_file()

    def downsample_cloud(self, cloud):
        _, indices = np.unique(
            np.round(cloud[:, :3], self.resolution),
            axis=0,
            return_index=True
        )
        return indices

    def rename_file(self):
        for room_folder in self.room_idxs:
            room_folder_path = os.path.join(self.root_dir, room_folder)
            for room_name in os.listdir(room_folder_path):
                # load np data
                room_path = os.path.join(room_folder_path, room_name)
                room_data = np.load(room_path)  # xyzrgbl, N*7
                # voxel downsample
                voxel_idxs = self.downsample_cloud(room_data)
                voxel_room_data = room_data[voxel_idxs]
                # add a new file path
                new_name = "Area_{}_{}".format(self.counter,room_name)
                old_name = room_path.split("/")[-1]
                new_name_path = os.path.join(room_path.strip(old_name), new_name)
                # rename file and save it
                #with open(new_name_path, "w") as f:
                np.savetxt(new_name_path, voxel_room_data)
                os.remove(room_path)

            self.counter += 1


if __name__ == "__main__":
    # add your folder path here
    file_path = (
        "/home/rical/catkin_ws/src/Pointnet_Pointnet2_pytorch/data/vkitti_indoor3d"
    )
    solution = KittiDatasetPrep(file_path)
