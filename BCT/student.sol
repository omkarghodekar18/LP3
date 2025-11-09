//SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;

contract StudentData{
    //structure
    struct student{
        string name;
        uint256 rollno;
    }

    //array
    student[] public studentarr;
    function addStudent(string memory name, uint rollno) public{
        for(uint i=0; i<studentarr.length; i++){
            if(studentarr[i].rollno == rollno){
                revert("Student with this roll no already exists");
            }
        }
        studentarr.push(student(name, rollno));
    }

    function getstudentlength() public view returns(uint){
        return studentarr.length;
    }

    function displayAllstudents() public view returns(student[] memory){
        return studentarr;
    }

    function getStudentByIndex(uint idx) public view returns(student memory){
        require(idx < studentarr.length, "index out of bound");
        return studentarr[idx];
    }

    //fallback
    fallback() external payable{
        //this function will handle external function calls that is not there in our contract
    }

    receive() external payable{
        //This function will handle the ether sent by extrenal user but without data mentioned
    }
}