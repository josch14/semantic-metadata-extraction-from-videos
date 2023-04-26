def ignore_video_id(gt, video_id, video_ids):
    # only use videos that are in both gt and submission
    if video_id not in gt.keys():
        return True

    # if there is no gt target, then skip this video
    if len(gt[video_id]) == 0:
        return True

    # use video_ids_restiction if given
    if video_ids != [] and video_id not in video_ids:
        return True

    return False


def get_checked_video_ids_for_eval(video_ids, submission, gt):
    # Calculate vieo_ids for which evaluation is possible
    checked_video_ids_for_eval = []
    for video_id in submission:
        if ignore_video_id(gt, video_id, video_ids):
            continue
        checked_video_ids_for_eval.append(video_id)
    return checked_video_ids_for_eval


def print_eval_info_different_TPs(TP_p, TP_gt, FP, FN):
    precision = TP_p / (TP_p + FP)
    recall = TP_gt / (TP_gt + FN)
    f1 = 2 * recall * precision / (recall + precision)
    print_measure("Precision", precision)
    print_measure("Recall", recall)
    print_measure("F1", f1)

    print()
    return precision, recall, f1


def print_eval_info(TP, FP, FN):
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * recall * precision / (recall + precision)
    print_measure("Precision", precision)
    print_measure("Recall", recall)
    print_measure("F1", f1)
    print()
    return precision, recall, f1


def print_measure(measure_name, value):
    str = "{:.2f}".format(round(value * 100, 2))
    print(f"{measure_name}: {str}")
