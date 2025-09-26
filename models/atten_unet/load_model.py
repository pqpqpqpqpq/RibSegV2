
import io
import torch
from cryptography.fernet import Fernet


def model_encryption(pth_file, encryp_file, license):
    """
    :param pth_file: 模型文件
    :param encryp_file: 加密后的模型文件
    :param license: 秘钥
    """
    checkpoint = torch.load(pth_file)
    b = io.BytesIO()
    torch.save(checkpoint, b)
    b.seek(0)
    pth_bytes = b.read()
    # key = Fernet.generate_key()
    encrypted_data = Fernet(license).encrypt(pth_bytes)
    with open(encryp_file, 'wb') as fw:
        fw.write(encrypted_data)


def model_decryption(encryt_file, license):
    with open(encryt_file, 'rb') as fr:
        encrypted_data = fr.read()

    decrypted_data = Fernet(license).decrypt(encrypted_data)
    b = io.BytesIO(decrypted_data)
    b.seek(0)
    checkpoint = torch.load(b)

    return checkpoint


if __name__ == "__main__":
    # rib_seg_key = Fernet.generate_key()
    rib_seg_key = "pzmjMg7vLv-ozVSwLOYGuYxonlOdLIoK3wpltP-ZfG0="
    orgin_ckpt = "../ckpts/rib_segmentor_cube64.pth"
    target_ckpt = "../ckpts/rib_segmentor_v0.2.220609.pth"
    model_encryption(pth_file=orgin_ckpt, encryp_file=target_ckpt, license=rib_seg_key)

