//SPDX-License-Identifier-UNLICENSED
pragma solidity ^0.8.0;

contract Bank{
    uint256 balance=0;
    address public accOwner;

    constructor(){
        accOwner = msg.sender;
    }

    function deposit() public payable{
        require(accOwner==msg.sender,"You are not the owner of this account");
        require(msg.value>0," anmount should be greater than 0");
        balance += msg.value;
    }
    
    function withdraw(uint256 amount) public payable{
        require(accOwner==msg.sender,"You are not the owner of this account");
        require(amount > 0," withdrawal money should be greater than 0");
        require(amount <= balance," Insufficient balance");
        balance-=amount;
        payable(msg.sender).transfer(amount);
    }

    function showbalance() public view returns(uint256){
        require(accOwner==msg.sender,"You are not the owner of this account");
        return balance;
    }
}