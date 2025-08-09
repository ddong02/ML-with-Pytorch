# fig = plt.figure(figsize=(16, 8.5))

# ax = fig.add_subplot(2, 5, 1)
# img, attr = celeba_train_dataset[0]
# ax.set_title('Crop to a \nbounding-box', size=15)
# ax.imshow(img)
# ax = fig.add_subplot(2, 5, 6)
# img_cropped = transforms.functional.crop(img, 50, 20, 128, 128)
# ax.imshow(img_cropped)

# ax = fig.add_subplot(2, 5, 2)
# img, attr = celeba_train_dataset[1]
# ax.set_title('Flip (horizontal)', size=15)
# ax.imshow(img)
# ax = fig.add_subplot(2, 5, 7)
# img_flipped = transforms.functional.hflip(img)
# ax.imshow(img_flipped)

# ax = fig.add_subplot(2, 5, 3)
# img, attr = celeba_train_dataset[2]
# ax.set_title('Adjust contrast', size=15)
# ax.imshow(img)
# ax = fig.add_subplot(2, 5, 8)
# img_adj_contrast = transforms.functional.adjust_contrast(
#     img, contrast_factor=2
# )
# ax.imshow(img_adj_contrast)

# ax = fig.add_subplot(2, 5, 4)
# img, attr = celeba_train_dataset[3]
# ax.set_title('Adjust brightness', size=15)
# ax.imshow(img)
# ax = fig.add_subplot(2, 5, 9)
# img_adj_brightness = transforms.functional.adjust_brightness(
#     img, brightness_factor=1.3
# )
# ax.imshow(img_adj_brightness)

# ax = fig.add_subplot(2, 5, 5)
# img, attr = celeba_train_dataset[4]
# ax.set_title('Center crop\nand resize', size=15)
# ax.imshow(img)
# ax = fig.add_subplot(2, 5, 10)
# img_center_crop = transforms.functional.center_crop(
#     img, [0.7*218, 0.7*178]
# )
# img_resized = transforms.functional.resize(
#     img_center_crop, size=(218, 178)
# )
# ax.imshow(img_resized)
# plt.show()

# torch.manual_seed(1)
# fig = plt.figure(figsize=(14,12))
# for i, (img, attr) in enumerate(celeba_train_dataset):
#     ax = fig.add_subplot(3, 4, i*4+1)
#     ax.imshow(img)
#     if i == 0:
#         ax.set_title('Orig.', size=15)
#     ax = fig.add_subplot(3, 4, i*4+2)
#     img_transform = transforms.Compose([
#         transforms.RandomCrop([178, 178])
#     ])
#     img_cropped = img_transform(img)
#     ax.imshow(img_cropped)
#     if i == 0:
#         ax.set_title('Step 1: Random Crop', size=15)
#     ax = fig.add_subplot(3, 4, i*4+3)
#     img_transform = transforms.Compose([
#         transforms.RandomHorizontalFlip()
#     ])
#     img_flip = img_transform(img_cropped)
#     ax.imshow(img_flip)
#     if i == 0:
#         ax.set_title('Step 2: Random Flip', size=15)
#     ax = fig.add_subplot(3, 4, i*4+4)
#     img_resized = transforms.functional.resize(
#         img_flip, size=(128, 128)
#     )
#     ax.imshow(img_resized)
#     if i == 0:
#         ax.set_title('Step 3: Resize', size=15)
#     if i == 2:
#         break
# plt.show()