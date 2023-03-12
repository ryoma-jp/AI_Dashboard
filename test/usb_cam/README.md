# USB Camera test program

This program is test program for USB Camera streaming and inferences.

## Usage

1. Prepare
    1. Create the categories list of COCO2017
        ```shell
        $ ./bin/create_categories_list_COCO2017.sh
        ```
2. Run ``usb_cam.py``
    ```shell:example
    $ python3 usb_cam.py --model_id 1
    ```

