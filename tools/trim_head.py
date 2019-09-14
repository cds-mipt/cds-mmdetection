import argparse
import torch


# model: [(weight_key, dimension), ...]
HEAD_WEIGHTS = {
    "cascade_rcnn": [
        ("bbox_head.0.fc_cls.weight", 0),
        ("bbox_head.0.fc_cls.bias", 0),
        ("bbox_head.1.fc_cls.weight", 0),
        ("bbox_head.1.fc_cls.bias", 0),
        ("bbox_head.2.fc_cls.weight", 0),
        ("bbox_head.2.fc_cls.bias", 0)
    ],
    "cascade_mask_rcnn": [
        ("bbox_head.0.fc_cls.weight", 0),
        ("bbox_head.0.fc_cls.bias", 0),
        ("bbox_head.1.fc_cls.weight", 0),
        ("bbox_head.1.fc_cls.bias", 0),
        ("bbox_head.2.fc_cls.weight", 0),
        ("bbox_head.2.fc_cls.bias", 0),

        ("mask_head.0.conv_logits.weight", 0),
        ("mask_head.0.conv_logits.bias", 0),
        ("mask_head.1.conv_logits.weight", 0),
        ("mask_head.1.conv_logits.bias", 0),
        ("mask_head.2.conv_logits.weight", 0),
        ("mask_head.2.conv_logits.bias", 0),
    ],
    "mask_rcnn": [
        ("bbox_head.fc_cls.weight", 0),
        ("bbox_head.fc_cls.bias", 0),
        ("mask_head.conv_logits.weight", 0),
        ("mask_head.conv_logits.bias", 0)
    ],
    "faster_rcnn": [
        ("bbox_head.fc_cls.weight", 0),
        ("bbox_head.fc_cls.bias", 0),
        ("bbox_head.fc_reg.weight", 0),
        ("bbox_head.fc_reg.bias", 0)
    ],
    "retinanet": [
        ("bbox_head.retina_cls.weight", 0),
        ("bbox_head.retina_cls.bias", 0)
    ]
}


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cp-in", type=str, required=True)
    parser.add_argument("--num-cls", type=int, required=True)
    parser.add_argument("--model", type=str, choices=HEAD_WEIGHTS.keys())
    parser.add_argument("--cp-out", type=str, required=True)
    return parser


def main(args):
    sd = torch.load(args.cp_in, map_location="cpu")
    sd["meta"]["config"] = sd["meta"]["config"].replace(
        "num_classes=81", "num_classes=" + str(args.num_cls)
    )
    # for k, v in sd["state_dict"].items():
    #     if "head" in k:
    #         print(k, v.shape)
    for k, v in HEAD_WEIGHTS[args.model]:
        dtype = sd["state_dict"][k].dtype
        shape = list(sd["state_dict"][k].shape)
        if args.model != "retinanet":
            shape[v] = args.num_cls * shape[v] // 81
        else:
            shape[v] = args.num_cls * shape[v] // 80
        # инициализация проводится в train.py после загрузки конфига
        sd["state_dict"][k] = torch.zeros(shape, dtype=dtype)
    torch.save(sd, args.cp_out)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
