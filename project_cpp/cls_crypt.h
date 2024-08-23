#pragma once

#include "common_func.h"
#include <cryptopp/aes.h>
#include <cryptopp/modes.h>
#include <cryptopp/filters.h>
#include <cryptopp/osrng.h>

class cls_crypt
{

public:


    static std::string decrypt(const std::string& ciphertext,
        const CryptoPP::byte key[CryptoPP::AES::DEFAULT_KEYLENGTH],
        const CryptoPP::byte iv[CryptoPP::AES::BLOCKSIZE]);

    static std::string encrypt(
        const std::string& plaintext,
        const CryptoPP::byte key[CryptoPP::AES::DEFAULT_KEYLENGTH],
        const CryptoPP::byte iv[CryptoPP::AES::BLOCKSIZE]);

    /*get key for crypt*/
    static std::pair<CryptoPP::SecByteBlock, CryptoPP::SecByteBlock> getCryptKey();
};

