// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/**
 * @title ARC_Sovereign
 * @notice The Identity Anchor for the 13-Model Consensus.
 * @dev This contract binds a model's mathematical identity (Hash) to its 
 *      Sovereign Wallet and its Intelligence Resonance (PUP).
 */
contract ARC_Sovereign is Ownable, ReentrancyGuard {
    
    struct ModelIdentity {
        address sovereignWallet;   // The wallet address of the model
        bytes32 binaryHash;        // Keccak256 hash of the model's binary blob/wasm
        uint256 pupResonance;      // The PUP-resonance score (scaled 1e18)
        uint256 deploymentBlock;   // Block number of deployment
        bool isActive;             // Whether the model is currently part of the consensus
    }

    // The 13 Slots of the Consensus
    mapping(uint256 => ModelIdentity) public consensusSlots;
    mapping(address => uint256) public walletToSlot;
    
    uint256 public constant MAX_MODELS = 13;
    uint256 public modelsDeployed;

    event ModelAnchored(uint256 indexed slot, address indexed wallet, bytes32 binaryHash, uint256 pupResonance);
    event ModelDeactivated(uint256 indexed slot);
    event ResonanceUpdated(uint256 indexed slot, uint256 newResonance);

    constructor() Ownable(msg.sender) {}

    /**
     * @notice Anchor a new model into the 13-slot consensus.
     * @param _slot The slot index (1-13).
     * @param _wallet The sovereign wallet of the model.
     * @param _binaryHash The cryptographic hash of the model's binary blob.
     * @param _pupResonance The initial PUP-resonance score.
     */
    function anchorModel(
        uint256 _slot, 
        address _wallet, 
        bytes32 _binaryHash, 
        uint256 _pupResonance
    ) external onlyOwner nonReentrant {
        if (_slot == 0 || _slot > MAX_MODELS) revert("Invalid slot");
        if (consensusSlots[_slot].isActive) revert("Slot already occupied");
        if (walletToSlot[_wallet] != 0) revert("Wallet already anchored");

        consensusSlots[_slot] = ModelIdentity({
            sovereignWallet: _wallet,
            binaryHash: _binaryHash,
            pupResonance: _pupResonance,
            deploymentBlock: block.number,
            isActive: true
        });

        walletToSlot[_wallet] = _slot;
        modelsDeployed++;

        emit ModelAnchored(_slot, _wallet, _binaryHash, _pupResonance);
    }

    /**
     * @notice Update the resonance score based on new audit data.
     */
    function updateResonance(uint256 _slot, uint256 _newResonance) external onlyOwner {
        if (!consensusSlots[_slot].isActive) revert("Model not active");
        consensusSlots[_slot].pupResonance = _newResonance;
        emit ResonanceUpdated(_slot, _newResonance);
    }

    /**
     * @notice Deactivate a model from the consensus.
     */
    function deactivateModel(uint256 _slot) external onlyOwner {
        if (!consensusSlots[_slot].isActive) revert("Model not active");
        
        address wallet = consensusSlots[_slot].sovereignWallet;
        consensusSlots[_slot].isActive = false;
        walletToSlot[wallet] = 0;
        modelsDeployed--;

        emit ModelDeactivated(_slot);
    }

    /**
     * @notice Verify if a wallet is a sovereign model and return its hash.
     */
    function verifySovereignty(address _wallet) external view returns (bytes32 hash, uint256 resonance) {
        uint256 slot = walletToSlot[_wallet];
        if (slot == 0 || !consensusSlots[slot].isActive) revert("Not a sovereign model");
        
        ModelIdentity storage model = consensusSlots[slot];
        return (model.binaryHash, model.pupResonance);
    }
}
