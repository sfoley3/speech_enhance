


def dtw_path(ref: np.ndarray, hyp: np.ndarray, scale_mode: str = "zscore"
             ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Joint multivariate DTW across F1/F2/F3 using librosa.sequence.dtw.
    `ref`, `hyp`: (T, 3) in raw Hz.  Returns aligned index arrays
    (i_ref, j_hyp) in time order (start -> end of vowel).
    """
    a = _rescale_for_dtw(ref, scale_mode)
    b = _rescale_for_dtw(hyp, scale_mode)
    # librosa expects (features, time)
    _, wp = librosa.sequence.dtw(
        X=a.T, Y=b.T,
        metric="euclidean",
        subseq=False,
        backtrack=True,
    )
    # wp is returned in reverse (end -> start); flip so it's start -> end
    wp = wp[::-1]
    return wp[:, 0], wp[:, 1]