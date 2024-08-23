#include "cls_crypt.h"

/*****************************************************************************/
/*encrypt*/
std::string cls_crypt::encrypt(
    const std::string& plaintext,
    const CryptoPP::byte key[CryptoPP::AES::DEFAULT_KEYLENGTH],
    const CryptoPP::byte iv[CryptoPP::AES::BLOCKSIZE]) 
{
    std::string ciphertext;

    try 
    {
        CryptoPP::CBC_Mode<CryptoPP::AES>::Encryption encryption(key, CryptoPP::AES::DEFAULT_KEYLENGTH, iv);
        CryptoPP::StringSource ss(plaintext, true,
            new CryptoPP::StreamTransformationFilter(encryption,
                new CryptoPP::StringSink(ciphertext)
            )
        );
    }
    catch (const CryptoPP::Exception& e) 
    {
        std::cerr << "Encryption error: " << e.what() << std::endl;
        exit(1);
    }

    return ciphertext;
}


/*****************************************************************************/
/*decrypt*/
std::string cls_crypt::decrypt(const std::string& ciphertext, 
    const CryptoPP::byte key[CryptoPP::AES::DEFAULT_KEYLENGTH],
    const CryptoPP::byte iv[CryptoPP::AES::BLOCKSIZE]) {
    std::string decryptedtext;

    try {
        CryptoPP::CBC_Mode<CryptoPP::AES>::Decryption decryption(key, CryptoPP::AES::DEFAULT_KEYLENGTH, iv);
        CryptoPP::StringSource ss(ciphertext, true,
            new CryptoPP::StreamTransformationFilter(decryption,
                new CryptoPP::StringSink(decryptedtext)
            )
        );
    }
    catch (const CryptoPP::Exception& e) {
        std::cerr << "Decryption error: " << e.what() << std::endl;
        exit(1);
    }

    return decryptedtext;
}

/*****************************************************************************/
/*get key for crypt*/
std::pair<CryptoPP::SecByteBlock, CryptoPP::SecByteBlock> cls_crypt::getCryptKey() 
{
    CryptoPP::AutoSeededRandomPool prng;

    CryptoPP::SecByteBlock key(CryptoPP::AES::DEFAULT_KEYLENGTH);
    CryptoPP::SecByteBlock iv(CryptoPP::AES::BLOCKSIZE);

    prng.GenerateBlock(key, key.size());
    prng.GenerateBlock(iv, iv.size());

    return std::make_pair(key, iv);
}