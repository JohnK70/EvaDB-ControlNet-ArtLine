import numpy as np
import pandas as pd

from evadb.catalog.catalog_type import NdArrayType
from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.decorators import forward, setup
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe
from evadb.utils.generic_utils import try_to_import_cv2


class BlurImage(AbstractFunction):
    @setup(cacheable=False, function_type="cv2-transformation", batchable=True)
    def setup(self):
        try_to_import_cv2()

    @property
    def name(self):
        return "blurImage"

    @forward(
        input_signatures=[
            PandasDataframe(
                columns=["data"],
                column_types=[NdArrayType.FLOAT32],
                column_shapes=[(None, None, 3)],
            )
        ],
        output_signatures=[
            PandasDataframe(
                columns=["blurframe"],
                column_types=[NdArrayType.FLOAT32],
                column_shapes=[(None, None)],
            )
        ],
    )
    def forward(self, frame: pd.DataFrame) -> pd.DataFrame:
        def BlurImage(row: pd.Series) -> np.ndarray:
            row = row.to_list()
            frame = row[0]

            import cv2

            frame = cv2.GaussianBlur(frame, (25, 25), 0)

            return frame

        ret = pd.DataFrame()
        ret["blurframe"] = frame.apply(BlurImage, axis=1)
        return ret