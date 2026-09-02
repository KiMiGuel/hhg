// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title FaceRegistry
 * @dev Stores tamper-evident records of face biometric fingerprints and
 *      discovered social media posts.
 */
contract FaceRegistry {
    struct VerificationRecord {
        bytes32 faceHash;        // SHA-256 biometric fingerprint of detected face
        string postUrl;          // Discovered social media / web post URL
        bytes32 dataHash;        // Combined SHA-256(faceHash + postUrl)
        uint256 blockTimestamp;  // On-chain recorded timestamp
        address registeredBy;    // Submitter wallet address
    }

    // Mapping: faceHash => VerificationRecord
    mapping(bytes32 => VerificationRecord) private records;

    // Events
    event RecordRegistered(
        bytes32 indexed faceHash,
        string postUrl,
        bytes32 dataHash,
        uint256 timestamp,
        address indexed registeredBy
    );

    /**
     * @notice Anchor a newly discovered social profile against a facial hash
     * @param _faceHash 32-byte cryptographic hash of the face
     * @param _postUrl The discovered social media URL
     * @param _dataHash Combined hash of face and post data
     */
    function registerRecord(
        bytes32 _faceHash,
        string calldata _postUrl,
        bytes32 _dataHash
    ) external {
        require(records[_faceHash].blockTimestamp == 0, "Record already exists for this face hash");

        records[_faceHash] = VerificationRecord({
            faceHash: _faceHash,
            postUrl: _postUrl,
            dataHash: _dataHash,
            blockTimestamp: block.timestamp,
            registeredBy: msg.sender
        });

        emit RecordRegistered(_faceHash, _postUrl, _dataHash, block.timestamp, msg.sender);
    }

    /**
     * @notice Re-verify data against on-chain storage
     * @param _faceHash 32-byte cryptographic hash of the face to query
     */
    function getRecord(bytes32 _faceHash)
        external
        view
        returns (
            bytes32 faceHash,
            string memory postUrl,
            bytes32 dataHash,
            uint256 timestamp,
            address registeredBy
        )
    {
        VerificationRecord memory rec = records[_faceHash];
        require(rec.blockTimestamp > 0, "No verification record found for this face hash");
        return (rec.faceHash, rec.postUrl, rec.dataHash, rec.blockTimestamp, rec.registeredBy);
    }
}
