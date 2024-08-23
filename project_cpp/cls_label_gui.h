#pragma once

#include "cls_base.h"
#include "cls_crypt.h"

class cls_label_gui
{
public:

	void saveLabelGui(
		const std::map<int, boost::uuids::uuid>& label_to_gui,
		const std::string& filename);

	void loadLabelGui(
		std::map<int, boost::uuids::uuid>& label_to_gui,
		const std::string& filename);

	/*save label and name to crypt file*/
	void saveLabelGuiCrypt(
		const std::map<int, boost::uuids::uuid>& label_to_gui,
		const std::string& filename);
	/*load label and name from crypt file*/
	void loadLabelGuiCrypt(
		std::map<int, boost::uuids::uuid>& label_to_gui,
		const std::string& filename);

	void setKeyForCrypt(const CryptoPP::byte key[CryptoPP::AES::DEFAULT_KEYLENGTH],
		const CryptoPP::byte iv[CryptoPP::AES::BLOCKSIZE]);


private:

	/*function for serialize map int - gui*/
	std::string serializeMap(const std::map<int, boost::uuids::uuid>& map);
	/*function for deserialize map int - gui*/
	std::map<int, boost::uuids::uuid> deserializeMap(const std::string& data);

	CryptoPP::byte _key[CryptoPP::AES::DEFAULT_KEYLENGTH];
	CryptoPP::byte _iv[CryptoPP::AES::BLOCKSIZE];
};

